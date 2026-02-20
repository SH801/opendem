import numpy as np
import os
import yaml
import signal
import sys
import time
# [2026-01-24] Always place imports at the top of the file
from osgeo import gdal, ogr, osr

# Standard GIS exception handling
gdal.UseExceptions()

class OpenDEM:
    def __init__(self, config_path):
        # Register Ctrl+C handler
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cache_dir = self.config.get('cache_dir', './cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        gdal.SetConfigOption('GDAL_WMS_CACHE_ENABLED', 'YES')
        gdal.SetConfigOption('GDAL_WMS_CACHE_DIR', self.cache_dir)
        gdal.SetConfigOption('GDAL_CACHEMAX', '2048')

        # Throttle threads to prevent AWS S3/DNS rate-limiting
        gdal.SetConfigOption('GDAL_NUM_THREADS', '2') 
        
        # Aggressive Retry Logic
        gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '20')
        gdal.SetConfigOption('GDAL_HTTP_RETRY_DELAY', '15')
        gdal.SetConfigOption('GDAL_HTTP_TIMEOUT', '60')
        gdal.SetConfigOption("GDAL_WMS_ZERO_BLOCK_HTTP_CODES", "0,500,503,404")
        gdal.SetConfigOption('GDAL_WMS_HTTP_ZEROBYTE_IS_ERROR', 'YES')

        self.log(f"Initialized opendem with config: {config_path}")

    def _handle_interrupt(self, sig, frame):
        self.log("Intercepted ctrl+C. Forcing exit...")
        os._exit(0)

    def log(self, message):
        print(f"[opendem] {message}")

    def _get_clipping_path(self):
        path = self.config.get('clipping')
        if path and path.startswith('http'):
            return f"/vsicurl/{path}"
        return path

    def _generate_vrt(self):
        vrt_path = os.path.join(self.cache_dir, "source.vrt")
        absolute_cache_path = os.path.abspath(self.cache_dir)
        
        vrt_content = f"""<GDAL_WMS>
    <Service name="TMS">
        <ServerUrl>{self.config['source']}</ServerUrl>
    </Service>
    <DataWindow>
        <UpperLeftX>-20037508.34</UpperLeftX>
        <UpperLeftY>20037508.34</UpperLeftY>
        <LowerRightX>20037508.34</LowerRightX>
        <LowerRightY>-20037508.34</LowerRightY>
        <TileLevel>15</TileLevel>
        <YOrigin>top</YOrigin>
    </DataWindow>
    <Projection>EPSG:3857</Projection>
    <BlockSizeX>256</BlockSizeX>
    <BlockSizeY>256</BlockSizeY>
    <BandsCount>3</BandsCount>
    <Cache>
        <Path>{absolute_cache_path}</Path>
        <Depth>2</Depth>
        <Extension>.tile</Extension>
        <Expires>-1</Expires>
        <CleanIndex>-1</CleanIndex>
        <MaxSize>50000</MaxSize>
    </Cache>
    <ZeroBlockHttpCodes>404,403,500,503</ZeroBlockHttpCodes>
    <ZeroBlockOnServerException>true</ZeroBlockOnServerException>
</GDAL_WMS>"""
        with open(vrt_path, "w") as f:
            f.write(vrt_content.strip())
        return vrt_path

    def progress_callback(self, complete, message, unknown):
        percent = int(complete * 100)
        if not hasattr(self, '_last_gdal_p'):
            self._last_gdal_p = -1
        if percent > self._last_gdal_p:
            self._last_gdal_p = percent
            if percent % 5 == 0:
                self.log(f"Progress: {percent}%")
        return 1

    def run(self):
        vrt_path = self._generate_vrt()
        temp_rgb = os.path.join(self.cache_dir, "temp_rgb.tif")
        
        max_retries = 5
        attempt = 0
        success = False

        while attempt < max_retries and not success:
            try:
                self.log(f"Warp Attempt {attempt + 1}/{max_retries}...")
                gdal.Warp(
                    temp_rgb,
                    vrt_path,
                    outputBounds=self.config['bounds'],
                    outputBoundsSRS="EPSG:4326",
                    xRes=self.config['resolution'],
                    yRes=self.config['resolution'],
                    dstSRS="EPSG:3857",
                    callback=self.progress_callback
                )
                success = True
            except RuntimeError as e:
                attempt += 1
                if "Could not resolve host" in str(e) or "IReadBlock failed" in str(e):
                    self.log(f"Network glitch: {e}. Retrying...")
                    time.sleep(10)
                else:
                    raise

        # 2. DECODE (WINDOWED to prevent 214GB RAM crash)
        self.log("Decoding RGB bands into metric elevation (tiled)...")
        ds_in = gdal.Open(temp_rgb)
        base_dem = os.path.join(self.cache_dir, "base_elevation.tif")
        
        # We create the skeleton of the base_dem first
        driver = gdal.GetDriverByName("GTiff")
        ds_out = driver.Create(base_dem, ds_in.RasterXSize, ds_in.RasterYSize, 1, gdal.GDT_Float32, options=['COMPRESS=DEFLATE', 'TILED=YES'])
        ds_out.SetProjection(ds_in.GetProjection())
        ds_out.SetGeoTransform(ds_in.GetGeoTransform())
        
        tile_size = 4096
        out_band = ds_out.GetRasterBand(1)
        
        for y in range(0, ds_in.RasterYSize, tile_size):
            rows = min(tile_size, ds_in.RasterYSize - y)
            for x in range(0, ds_in.RasterXSize, tile_size):
                cols = min(tile_size, ds_in.RasterXSize - x)
                
                r = ds_in.GetRasterBand(1).ReadAsArray(x, y, cols, rows).astype(np.float32)
                g = ds_in.GetRasterBand(2).ReadAsArray(x, y, cols, rows).astype(np.float32)
                b = ds_in.GetRasterBand(3).ReadAsArray(x, y, cols, rows).astype(np.float32)
                
                elevation = (r * 256.0 + g + b / 256.0) - 32768.0
                out_band.WriteArray(elevation, x, y)
        
        # Flush and close to free resources for the next step
        ds_out.FlushCache()
        ds_in = ds_out = None

        # 3. PROCESS & CLIP
        self._execute_process(base_dem)

    def _save_raster(self, data, source_ds, path, nodata=None, dtype=gdal.GDT_Float32):
        """Standard saver for smaller arrays (kept for logic compatibility)."""
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(path, source_ds.RasterXSize, source_ds.RasterYSize, 1, dtype, options=['COMPRESS=DEFLATE'])
        out_ds.SetProjection(source_ds.GetProjection())
        out_ds.SetGeoTransform(source_ds.GetGeoTransform())
        band = out_ds.GetRasterBand(1)
        if nodata is not None:
            band.SetNoDataValue(nodata)
        band.WriteArray(data)
        out_ds.FlushCache()
        out_ds = None

    def _save_as_vector(self, src_path, output_path):
        """Converts mask raster to GPKG using tiled polygonize."""
        ds = gdal.Open(src_path)
        band = ds.GetRasterBand(1)
        
        vec_driver = ogr.GetDriverByName("GPKG")
        if os.path.exists(output_path):
            vec_driver.DeleteDataSource(output_path)
            
        out_ds = vec_driver.CreateDataSource(output_path)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        
        layer = out_ds.CreateLayer("mask", srs, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("dn", ogr.OFTInteger))

        gdal.Polygonize(band, band, layer, 0, [], callback=self.progress_callback)
        out_ds = ds = None

    def _execute_process(self, dem_path):
        process_type = self.config.get('process')
        output_name = self.config.get('output')
        clipping_path = self._get_clipping_path()
        mask_cfg = self.config.get('mask')
        nodata_val = -9999

        self.log(f"Running terrain analysis: '{process_type}'...")
        temp_proc = os.path.join(self.cache_dir, "temp_proc.tif")
        # Potentially replace with:
        # smoothed_dem = gdal.Warp('', dem_path, format='MEM', resampleAlg='cubicspline')
        # gdal.DEMProcessing(temp_proc, smoothed_dem, process_type, alg='ZevenbergenThorne')
        gdal.DEMProcessing(temp_proc, dem_path, process_type, alg='ZevenbergenThorne')

        if clipping_path:
            self.log(f"Applying cutline: {clipping_path}")
            process_source = os.path.join(self.cache_dir, "final_clipped.tif")
            gdal.Warp(process_source, temp_proc, cutlineDSName=clipping_path, 
                      cropToCutline=True, dstNodata=nodata_val)
        else:
            process_source = temp_proc

        # Handle Masks / Thresholding via windowed logic to avoid MemoryError
        if mask_cfg:
            self.log(f"Mask detected. Generating binary output...")
            mask_tif = os.path.join(self.cache_dir, "temp_mask.tif")
            ds_p = gdal.Open(process_source)
            
            drv = gdal.GetDriverByName("GTiff")
            ds_m = drv.Create(mask_tif, ds_p.RasterXSize, ds_p.RasterYSize, 1, gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
            ds_m.SetProjection(ds_p.GetProjection())
            ds_m.SetGeoTransform(ds_p.GetGeoTransform())
            
            tile_size = 4096
            b_in = ds_p.GetRasterBand(1)
            b_out = ds_m.GetRasterBand(1)
            
            for y in range(0, ds_p.RasterYSize, tile_size):
                rows = min(tile_size, ds_p.RasterYSize - y)
                for x in range(0, ds_p.RasterXSize, tile_size):
                    cols = min(tile_size, ds_p.RasterXSize - x)
                    data = b_in.ReadAsArray(x, y, cols, rows)
                    
                    condition = np.ones(data.shape, dtype=bool)
                    if 'min' in mask_cfg: condition &= (data >= mask_cfg['min'])
                    if 'max' in mask_cfg: condition &= (data <= mask_cfg['max'])
                    
                    final_mask = np.where(condition & (data != nodata_val), 1, 0).astype(np.uint8)
                    b_out.WriteArray(final_mask, x, y)
            
            ds_m.FlushCache()
            ds_p = ds_m = None
            final_proc_path = mask_tif
            current_nodata = 0
        else:
            final_proc_path = process_source
            current_nodata = nodata_val

        # 4. Final Export Logic
        if output_name.lower().endswith('.gpkg'):
            self.log(f"Exporting to Vector: {output_name}")
            self._save_as_vector(final_proc_path, output_name)
        else:
            self.log(f"Exporting to Raster: {output_name}")
            # If we didn't mask, we just move the continuous file to output
            if not mask_cfg:
                if os.path.exists(output_name): os.remove(output_name)
                os.rename(final_proc_path, output_name)
            else:
                # If we masked, the file is already a Byte GeoTIFF at mask_tif
                if os.path.exists(output_name): os.remove(output_name)
                os.rename(final_proc_path, output_name)

        self.log(f"Process complete: {output_name}")
        
def main():
    if len(sys.argv) < 2:
        print("Usage: opendem <config.yml>")
        sys.exit(1)
    app = OpenDEM(sys.argv[1])
    app.run()

if __name__ == "__main__":
    main()