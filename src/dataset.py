import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import box

from PIL import Image

class SatelliteRoadDataset(Dataset):
    def __init__(self, roads_data, images_paths, patch_size, transform=None, rotation=None, stretch_factor=None, buffer_size=0):
        self.roads_data = roads_data
        self.images_paths = images_paths
        self.patch_size = patch_size
        self.transform = transform
        self.rotation = rotation
        self.stretch_factor = stretch_factor
        self.buffer_size = buffer_size

        # Calculate the number of patches (all bands have the same dimensions)
        with rasterio.open(self.images_paths[0]) as src:
            self.img_width, self.img_height = src.width, src.height
            self.src_transform = src.transform
        self.num_patches_x = self.img_width // self.patch_size
        self.num_patches_y = self.img_height // self.patch_size
        self.total_patches = self.num_patches_x * self.num_patches_y

    def __len__(self):
        return self.total_patches

    def __getitem__(self, index):
        # Calculate patch coordinates
        patch_x = (index % self.num_patches_x) * self.patch_size
        patch_y = (index // self.num_patches_x) * self.patch_size

        # Extract the satellite image patch
        patch_data = np.empty((len(self.images_paths), self.patch_size, self.patch_size), dtype=np.float32)
        for i, path in enumerate(self.images_paths):
            with rasterio.open(path) as src:
                patch_data[i, :, :] = src.read(1, window=Window(patch_x, patch_y, self.patch_size, self.patch_size))

        # Rasterize the road data for the corresponding patch
        label_patch = self.rasterize_roads(patch_x, patch_y)

        # Normalize the satellite image patch
        for i in range(patch_data.shape[0]):
            patch_data[i, :, :] = self.normalize(patch_data[i, :, :])

        # Apply transformations if any
        if self.transform:
            patch_data = self.transform(patch_data)
            label_patch = self.transform(label_patch)

        # Apply roation if any
        if self.rotation:
            rotation_angle = 90
            patch_data, label_patch = self.rotate_patches(patch_data, label_patch, rotation_angle)

        # Apply stretch if any
        if self.stretch_factor:
            patch_data, label_patch = self.stretch_image(patch_data, label_patch, self.stretch_factor)

        return torch.from_numpy(patch_data), torch.from_numpy(label_patch)

    def rotate_patches(self, patch_data, label_patch, rotation_angle):
        # Rotate the satellite image patch
        rotated_patch_data = []
        for i in range(patch_data.shape[0]):
            rotated_band = transforms.functional.to_pil_image(patch_data[i, :, :])
            rotated_band = transforms.functional.rotate(rotated_band, rotation_angle)
            rotated_band = np.array(rotated_band)
            rotated_patch_data.append(rotated_band)

        rotated_patch_data = np.stack(rotated_patch_data)

        # Rotate the road label patch
        rotated_label_patch = transforms.functional.to_pil_image(label_patch)
        rotated_label_patch = transforms.functional.rotate(rotated_label_patch, rotation_angle)
        rotated_label_patch = np.array(rotated_label_patch)

        return rotated_patch_data, rotated_label_patch

    def stretch_image(self, patch_data,label_patch, stretch_factor):

        num_bands, height, width = patch_data.shape
        stretched_patch = np.zeros((num_bands, width, int(height *stretch_factor)), dtype=patch_data.dtype)

        # Stretch the satellite image patch
        for i in range(patch_data.shape[0]):

            patch_data_2d = patch_data[i, :, :]
            image_pil = Image.fromarray(patch_data_2d)
            new_width = int(image_pil.width * stretch_factor)
            stretched_image = image_pil.resize((new_width, image_pil.height), Image.BILINEAR)
            stretched_image_array = np.array(stretched_image)
            stretched_patch[i,:,:] = stretched_image_array

        # Stretch the label patch
        image_pil = Image.fromarray(label_patch)
        new_width = int(image_pil.width * stretch_factor)
        stretched_image = image_pil.resize((new_width, image_pil.height), Image.BILINEAR)
        stretched_label_patch_array = np.array(stretched_image)
            
        return stretched_patch, stretched_label_patch_array

    def normalize(self, band):
        # Min-max normalization
        min = band.min()
        max = band.max()
        if max - min == 0:
            return np.zeros(band.shape)
        else:
            return (band - min) / (max - min)

    def rasterize_roads(self, patch_x, patch_y):
        # Define the bounds of the raster patch
        bounds = rasterio.windows.bounds(Window(patch_x, patch_y, self.patch_size, self.patch_size), self.src_transform)

        # Create a transformation for the patch
        patch_transform = from_bounds(*bounds, self.patch_size, self.patch_size)

        # Clip the road data to the patch bounds
        clipped_roads = self.roads_data.clip(box(*bounds))

        # Add a buffer to the clipped road data
        clipped_roads.geometry = clipped_roads.buffer(self.buffer_size)

        # Check if there are any road geometries in the patch
        if clipped_roads.empty:
            return np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)

        # Rasterize the clipped road geometries
        road_mask = rasterize(
            shapes=clipped_roads.geometry,
            out_shape=(self.patch_size, self.patch_size),
            transform=patch_transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )

        return road_mask
