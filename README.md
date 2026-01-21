# opendem - library for processing Digital Elevation Model (DEM) files

## Overview
The `opendem` library provides a simple command line interface for automatically retrieving RGB Terrain tiles from a remote data repository, amalgamating the tiles and then processing the result to produce GeoTIFF or GKPG files. 

A `yml` configuration file is used to provide key parameters to the library.

## Key features

- Uses GDAL to manage caching so repeated runs at same resolution use locally cached files.
- Run simple processing through commands like `slope` or `aspect`.
- Clip to a user-supplied outline.
- Vectorize raster bitmap by applying min/max mask and setting output file extension to `.gpkg`.

## Installation

```
pip install git+https://
```

To use the library, enter:

```
opendem /path/to/conf.yml
```

## Configuration file

The `.yml` configuration file should have the following format:

```
# ----------------------------------------------------
# sample.yml
# Sample yml configuration file
# ----------------------------------------------------

# Link to this GitHub code repository 
# This can be used to host yml files on an open data server and automatically install the library just-in-time
codebase:
  https://github.com/

# Link to Mapzen Terrarium bucket on AWS (RGB-encoded PNGs)
source:
  https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png

# Directory where downloaded tiles and temporary data are stored
cache_dir:
  ./tile_cache

# Bounding box in WGS84 coordinates: [min_lon, min_lat, max_lon, max_lat]
bounds:
  [-9.0, 49.0, 2.0, 61.0]

# The horizontal pixel size of the output raster in meters
resolution:
  20

# External URL or path to a geometry file used to crop the output to a specific shape
clipping:
  https://github.com/open-wind/openwindenergy/raw/refs/heads/main/overall-clipping.gpkg

# The GDAL operation to run (e.g., slope, aspect, hillshade, roughness)
process:
  slope

# The exact name and extension of the final file generated
output:
  slope-too-steep--uk.gpkg

# Filters the final result values, e.g. degrees for slope
mask:
  # Minimum excessive slope is 5 degrees
  min: 
    5.001
  
  # Maxium excessive slope is 90 degrees
  max: 
    100
```

## Possible uses

- Generating slope and aspect vector files for solar farm mapping.
- Generating area-specific DEM heightmaps.
- Conveniently generate small DEM heightmaps for web applications such as wind 'viewshed' analysis.

