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
    total_pixels = 0
    for i in range(len(dataset)):
        label_mask = dataset[i][1]
        if torch.is_tensor(label_mask):
            label_mask = label_mask.numpy()
        road_pixels += np.count_nonzero(label_mask)
        total_pixels += label_mask.size
    return torch.tensor([total_pixels / road_pixels - 1])