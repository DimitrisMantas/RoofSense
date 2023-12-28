# TODO

- [X] Install labelling software and play around with real-world data.
    - GeoTIFFs with more than three bands are not supported.
- [] Finalize the problem statement and research questions.
- [] Make a flowchart of the preprocessing pipeline.
- [X] See what is left to be done to finalize the preprocessing pipeline.
- [] Check CEGM2003 Unit 2 lectures.

## Misc

- [] Structure the completed interviews.

# VRT Generation

1. ~~Fetch all relevant data.~~
2. ~~Buffer the building footprints.~~
3. ~~Clip the data to the extents of the
   buffered footprints.~~
    - This operation must not overwrite
      the original files since they will
      most likely be required to process
      more than a single tile.
4. *Rasterize the remaining LiDAR data.*
5. *Merge all similar rasters.*
    - TODO: Use the profile of the BM5
      imagery.
    - TODO: Check if overlapping rasters
      can be merged.
6. Generate the VRT.
7. Tile the VRT using GDAL.