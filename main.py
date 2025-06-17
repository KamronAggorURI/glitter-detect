'''
This program will sample a hsv value from a specified pixel in an image,
then segment the image into patches based on the sampled HSV value.
and save the patches to a specified directory.'''

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops, label

def segment_image(
        image,
        n_segments=100,
        min_area=100,
        aspect_range=(0.5, 2.0)
):
    segments = slic(image, n_segments=n_segments, compactness=10, sigma=1, start_label=1)
    mask = np.zeros(segments.shape, dtype=bool)

    for region in regionprops(label(segments)):
        if region.area < min_area:
            continue
        aspect_ratio = region.bbox[2] / region.bbox[3]
        if not (aspect_range[0] <= aspect_ratio <= aspect_range[1]):
            continue
        mask[segments == region.label] = True

    return mask

def sample_hsv(image):
    """
    Display an image and let the user click a point to sample its HSV value.
    Returns HSV in unit scale (0-1, 0-1, 0-1).
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title("Click a pixel to sample HSV")
    plt.axis('off')
    plt.tight_layout()

    # Capture a point from user click
    points = plt.ginput(1, timeout=0, show_clicks=True)
    if points:
        x, y = int(points[0][0]), int(points[0][1])
        hsv = rgb2hsv(image)
        h, s, v = hsv[y, x]
        print(f'H={h:.2f}, S={s:.2f}, V={v:.2f} (unit scale)')
        plt.show()
        return h, s, v
    else:
        print("No point selected.")
        plt.show()
        return 0.0, 0.0, 0.0

def extract_patches(image, mask):
    labeled_mask = label(mask)
    patches = []
    for region in regionprops(labeled_mask):
        minr, minc, maxr, maxc = region.bbox
        patch = image[minr:maxr, minc:maxc]
        patches.append(patch)
    return patches

def save_patches(patches, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, patch in enumerate(patches):
        patch_path = os.path.join(output_dir, f'patch_{i}.png')
        cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

def main(image_path, output_dir, n_segments=100, min_area=100, aspect_range=(0.5, 2.0)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv_value = sample_hsv(image)
    print(f'Sampled HSV value: {hsv_value}')

    mask = segment_image(image, n_segments=n_segments, min_area=min_area, aspect_range=aspect_range)
    patches = extract_patches(image, mask)

    save_patches(patches, output_dir)
    print(f'Saved {len(patches)} patches to {output_dir}')

if __name__ == "__main__":
    main(
        'datasets/Orange Glitter/Orange2Week30umTest.png', 'output/patches',
        n_segments=100, min_area=100, aspect_range=(0.5, 2.0)
    )