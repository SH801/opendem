import numpy as np
import os
import yaml
import signal
import sys
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
        
        gdal.SetConfigOption('GDAL_HTTP_CACHE', 'YES')
        gdal.SetConfigOption('GDAL_HTTP_CACHE_DIRECTORY', self.cache_dir)
        
        self.log(f"Initialized opendem with config: {config_path}")

    def _handle_interrupt(self, sig, frame):
        self.log("Intercepted ctrl+C. Forcing exit...")
        os._exit(0)

    def log(self, message):
        """
        Easily overridable logging function. 
        You could swap this out for a logger or print to a file.
        """
        print(f"[opendem] {message}")

    def _get_clipping_path(self):
        path = self.config.get('clipping')
        if path and path.startswith('http'):
            return f"/vsicurl/{path}"
        return path

    def _generate_vrt(self):
        vrt_path = os.path.join(self.cache_dir, "source.vrt")
        absolute_cache_path = os.path.abspath(self.cache_dir)
        
        # Note: 256px block size fixed for S3 Terrarium tiles
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
    <Cache><Path>{absolute_cache_path}</Path></Cache>
</GDAL_WMS>"""
        with open(vrt_path, "w") as f:
            f.write(vrt_content.strip())
        return vrt_path

    def progress_callback(self, complete, message, unknown):
        percent = int(complete * 100)
        
        # Initialize the attribute on the instance (self) if it doesn't exist
        if not hasattr(self, '_last_gdal_p'):
            self._last_gdal_p = -1
            
        # Only log when the percentage actually increments
        if percent > self._last_gdal_p:
            self._last_gdal_p = percent
            # Log every 5% to keep the UI snappy and avoid database bloat
            if percent % 5 == 0:
                self.log(f"[opendem] Warp Progress: {percent}%")
                
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
                    self.log(f"Network glitch detected: {e}")
                    if attempt < max_retries:
                        self.log("Retrying in 10 seconds...")
                        import time
                        time.sleep(10)
                    else:
                        self.log("Max retries reached. Check your internet connection.")
                        raise
                else:
                    raise # If it's a different error (like Disk Full), stop immediately

        # 2. DECODE
        self.log("Decoding RGB bands into metric elevation data...")
        ds = gdal.Open(temp_rgb)
        r = ds.GetRasterBand(1).ReadAsArray().astype(float)
        g = ds.GetRasterBand(2).ReadAsArray().astype(float)
        b = ds.GetRasterBand(3).ReadAsArray().astype(float)
        
        elevation = (r * 256.0 + g + b / 256.0) - 32768.0
        self.log(f"Elevation stats: Min {np.min(elevation):.2f}m, Max {np.max(elevation):.2f}m")

        base_dem = os.path.join(self.cache_dir, "base_elevation.tif")
        self._save_raster(elevation, ds, base_dem)
        
        # 3. PROCESS & CLIP
        self._execute_process(base_dem)
        ds = None

    def _save_raster(self, data, source_ds, path, nodata=None):
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(path, source_ds.RasterXSize, source_ds.RasterYSize, 1, gdal.GDT_Float32)
        out_ds.SetProjection(source_ds.GetProjection())
        out_ds.SetGeoTransform(source_ds.GetGeoTransform())
        
        band = out_ds.GetRasterBand(1)
        if nodata is not None:
            band.SetNoDataValue(nodata)
            
        band.WriteArray(data)
        out_ds.FlushCache()
        out_ds = None

    def _save_as_vector(self, data, source_ds, output_path):
        """Converts binary raster data to a GeoPackage multipolygon."""
        
        # Save binary data to a temporary memory raster first
        mem_driver = gdal.GetDriverByName('MEM')
        tmp_ds = mem_driver.Create('', source_ds.RasterXSize, source_ds.RasterYSize, 1, gdal.GDT_Byte)
        tmp_ds.SetProjection(source_ds.GetProjection())
        tmp_ds.SetGeoTransform(source_ds.GetGeoTransform())
        band = tmp_ds.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(0)

        # Create the GPKG
        vec_driver = ogr.GetDriverByName("GPKG")
        if os.path.exists(output_path):
            vec_driver.DeleteDataSource(output_path)
            
        out_datasource = vec_driver.CreateDataSource(output_path)
        srs = ogr.osr.SpatialReference()
        srs.ImportFromWkt(source_ds.GetProjection())
        
        layer = out_datasource.CreateLayer("mask", srs, ogr.wkbPolygon)
        fd = ogr.FieldDefn("dn", ogr.OFTInteger) # dn=1 for the mask area
        layer.CreateField(fd)

        # Polygonize: Only pixels with value 1 are converted
        gdal.Polygonize(band, band, layer, 0, [], callback=self.progress_callback)
        
        # # Final cleanup: Remove the features where DN=0 (if any created)
        # layer.SetAttributeFilter("dn = 0")
        # for feat in layer:
        #     layer.DeleteFeature(feat.GetFID())
            
        out_datasource = None

    def _execute_process(self, dem_path):
        process_type = self.config.get('process')
        output_name = self.config.get('output')
        clipping_path = self._get_clipping_path()
        mask_cfg = self.config.get('mask')  # None if not set
        nodata_val = -9999

        self.log(f"Running terrain analysis: '{process_type}'...")
        temp_proc = os.path.join(self.cache_dir, "temp_proc.tif")
        
        # 1. Process on the full rectangle for edge accuracy
        gdal.DEMProcessing(temp_proc, dem_path, process_type)

        # 2. Apply clipping
        if clipping_path:
            self.log(f"Applying final cutline: {clipping_path}")
            process_source = os.path.join(self.cache_dir, "final_clipped.tif")
            gdal.Warp(process_source, temp_proc, cutlineDSName=clipping_path, 
                      cropToCutline=True, dstNodata=nodata_val)
        else:
            process_source = temp_proc

        ds_proc = gdal.Open(process_source)
        data = ds_proc.GetRasterBand(1).ReadAsArray().astype(float)

        # 3. Decision Logic: Continuous vs Binary
        if mask_cfg:
            self.log(f"Mask detected. Generating binary output (Thresholds: {mask_cfg})")
            # Create a boolean mask based on thresholds
            condition = np.ones(data.shape, dtype=bool)
            if 'min' in mask_cfg:
                condition &= (data >= mask_cfg['min'])
            if 'max' in mask_cfg:
                condition &= (data <= mask_cfg['max'])
            
            # Convert to Binary (1 for True, 0 for False/NoData)
            final_data = np.where(condition & (data != nodata_val), 1, 0).astype(np.uint8)
            current_nodata = 0 
        else:
            self.log("No mask detected. Generating continuous float output.")
            final_data = data
            current_nodata = nodata_val

        # 4. Decision Logic: GeoTIFF vs GPKG
        if output_name.lower().endswith('.gpkg'):
            self.log(f"Exporting to Vector format: {output_name}")
            self._save_as_vector(final_data, ds_proc, output_name)
        else:
            self.log(f"Exporting to Raster format: {output_name}")
            # Use Byte for binary, Float32 for continuous
            dtype = gdal.GDT_Byte if mask_cfg else gdal.GDT_Float32
            self._save_raster(final_data, ds_proc, output_name, nodata=current_nodata, dtype=dtype)

        self.log(f"Process complete: {output_name}")
        
def main():
    import sys
    
    # Check if a config file was provided
    if len(sys.argv) < 2:
        print("Usage: opendem <config.yml>")
        sys.exit(1)

    config_path = sys.argv[1]
    
    # Check if the file actually exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # Initialize and run
    app = OpenDEM(config_path)
    app.run()

if __name__ == "__main__":
    main()