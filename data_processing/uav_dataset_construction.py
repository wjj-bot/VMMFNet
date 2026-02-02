"""
Filename: uav_dataset_construction.py
Description: Automated pipeline for multispectral/RGB image tiling,
             normalization, and multi-modal alignment for PWD detection.
Author: [Your Name/Lab Name]
"""

import cv2
import os
import numpy as np


class UAVDatasetBuilder:
    def __init__(self, tile_size=640, stride=640):
        self.tile_size = tile_size
        self.stride = stride

    def normalize_multispectral(self, img):
        """Perform min-max normalization for multispectral bands."""
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)

    def process_orthomosaic(self, rgb_path, ms_path, output_dir):
        """
        Slices large mosaics into smaller patches while ensuring
        RGB and Multispectral images are spatially aligned.
        """
        print(f"Processing: {os.path.basename(rgb_path)}")

        # Load images (using OpenCV as an example)
        rgb_img = cv2.imread(rgb_path)
        ms_img = cv2.imread(ms_path, cv2.IMREAD_UNCHANGED)  # Load multispectral

        h, w, _ = rgb_img.shape

        count = 0
        for y in range(0, h - self.tile_size, self.stride):
            for x in range(0, w - self.tile_size, self.stride):
                # Crop patches
                rgb_tile = rgb_img[y:y + self.tile_size, x:x + self.tile_size]
                ms_tile = ms_img[y:y + self.tile_size, x:x + self.tile_size]

                # Save processed tiles
                tile_name = f"tile_{y}_{x}.png"
                cv2.imwrite(os.path.join(output_dir, "rgb", tile_name), rgb_tile)
                np.save(os.path.join(output_dir, "ms", tile_name.replace('.png', '.npy')), ms_tile)
                count += 1

        print(f"Construction complete. {count} aligned tiles generated.")


if __name__ == "__main__":
    # Initialize builder with specific tile size
    builder = UAVDatasetBuilder(tile_size=640, stride=512)
    # Run the pipeline
    # builder.process_orthomosaic('path/to/rgb.tif', 'path/to/ms.tif', './output')