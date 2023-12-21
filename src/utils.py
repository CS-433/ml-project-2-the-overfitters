import numpy as np
import torch
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Function to project the landsat image to Web Mercator (EPSG: 3395)
def reproject_landsat_image(landsat_image_path, output_path):
    with rasterio.open(landsat_image_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:3395', src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'EPSG:3395',
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:3395',
                    resampling=Resampling.nearest)
                
def compute_class_weights(dataset):
    road_pixels = 0
    non_road_pixels = 0
    for i in range(len(dataset)):
        label_mask = dataset[i][1]
        road_pixels += torch.sum(label_mask == 1)
        non_road_pixels += torch.sum(label_mask == 0)
    return torch.tensor([non_road_pixels/road_pixels])

    