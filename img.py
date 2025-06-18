import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, mark_boundaries
from skimage.util import img_as_float
from skimage.color import rgb2gray, label2rgb, rgb2hsv, hsv2rgb
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects, binary_opening, disk
import os
import glob

def hypersegmentation(img, method='felzenszwalb', **kwargs):
    """
    Perform hypersegmentation on an image using specified method.
    
    Parameters:
    - img: Input image (numpy array).
    - method: Segmentation method to use ('felzenszwalb' or 'slic').
    - kwargs: Additional parameters for the segmentation method.
    
    Returns:
    - segments: Segmented image as a numpy array."""
    img_float = img_as_float(img)
    if method == 'felzenszwalb':
        segments = felzenszwalb(img_float, **kwargs)
        print(f'Felzenszwalb number of segments: {len(np.unique(segments))}')
    elif method == 'slic':
        segments = slic(img_float, **kwargs)
        print(f'SLIC number of segments: {len(np.unique(segments))}')
    else:
        raise ValueError("Unsupported segmentation method: {}".format(method))
    return segments

def hsv_deg_to_unit(h, s, v):
    """Convert HSV from (deg, %, %) to (0-1, 0-1, 0-1)"""
    return h / 360.0, s / 100.0, v / 100.0

def filter_segments(
    image, segments, min_area=10, aspect_range=(0.2, 5.0),
    min_brightness=0.00, max_brightness=1.00,
    target_rgb=None, rgb_tolerance=0.25,  # Euclidean distance in RGB [0,1]
    max_eccentricity=0.60, min_color_fraction=0
):
    labels = label(segments)
    props = regionprops(labels)
    mask = np.zeros_like(segments, dtype=bool)
    gray = rgb2gray(image)

    valid_indices = []
    for prop in props:
        area = prop.area
        y0, x0, y1, x1 = prop.bbox
        aspect_ratio = (x1 - x0 + 1) / (y1 - y0 + 1)
        segment_mask = (labels == prop.label)
        mean_brightness = gray[segment_mask].mean()
        eccentricity = prop.eccentricity

        # Color filtering in RGB
        if target_rgb is not None:
            rgb_pixels = image[segment_mask]
            dists = np.linalg.norm(rgb_pixels - target_rgb, axis=1)
            color_pixels = (dists <= rgb_tolerance)
            color_fraction = color_pixels.sum() / rgb_pixels.shape[0]
        else:
            color_fraction = 1.0

        if (
            (area >= min_area) and 
            (aspect_range[0] <= aspect_ratio <= aspect_range[1]) and 
            (min_brightness <= mean_brightness <= max_brightness) and
            (color_fraction >= min_color_fraction) and
            (eccentricity <= max_eccentricity)
        ):
            valid_indices.append(prop.label)
    if valid_indices:
        mask[np.isin(labels, valid_indices)] = True

    mask = binary_opening(mask, disk(1))
    mask = remove_small_objects(mask, min_size=30)

    return mask * segments

def extract_segment_patches(image, segments):
    patches = []
    for seg_val in np.unique(segments):
        if seg_val == 0:  # Skip background
            continue
        mask = (segments == seg_val)
        if mask.sum() < 100:
            continue
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        patch = image[y0:y1+1, x0:x1+1]
        patches.append(patch)
    return patches

def show_segments(img, segments):
    plt.figure(figsize=(10, 10))
    plt.imshow(mark_boundaries(img, segments))
    plt.axis('off')
    plt.title('Segmented Image with Boundaries')
    plt.tight_layout()
    plt.show()

def show_segmented_patches(patches):
    for i, patch in enumerate(patches):
        plt.figure(figsize=(4, 4))
        plt.imshow(patch)
        plt.axis('off')
        plt.title(f'Patch {i+1}')
        plt.tight_layout()
        plt.show()
        if i == 19:
            break

def save_patches(patches, save_dir, image_idx):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, patch in enumerate(patches):
        patch_file = os.path.join(save_dir, f'patch_{image_idx}_{i+1}.png')
        plt.imsave(patch_file, patch)
        print(f'Saved patch {i+1} to {patch_file}')

# Sample and print the HSV value of a specified pixel
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
    

def saturate_image(image, saturation_factor=1.5):
    """
    Increase the saturation of an image.
    Parameters:
    - image: Input image (numpy array).
    - saturation_factor: Factor by which to increase saturation.
    
    Returns:
    - Saturated image (numpy array).
    """
    hsv = rgb2hsv(image)
    hsv[..., 1] *= saturation_factor
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)  # Ensure S is in [0, 1]
    hsv = hsv2rgb(hsv)
    return hsv

def sample_rgb(image):
    """
    Display an image and let the user click a point to sample its RGB value.
    Returns RGB in unit scale (0-1, 0-1, 0-1).
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title("Click a pixel to sample RGB")
    plt.axis('off')
    plt.tight_layout()

    points = plt.ginput(1, timeout=0, show_clicks=True)
    if points:
        x, y = int(points[0][0]), int(points[0][1])
        rgb = image[y, x]
        print(f'R={rgb[0]:.2f}, G={rgb[1]:.2f}, B={rgb[2]:.2f} (unit scale)')
        plt.show()
        return rgb
    else:
        print("No point selected.")
        plt.show()
        return np.array([0.0, 0.0, 0.0])


# --------- main --------

image_extensions = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
image_paths = sorted(
    [file for ext in image_extensions for file in glob.glob(f"datasets/Orange Glitter/{ext}", recursive=True)]
)

for idx, image_path in enumerate(image_paths, 1):
    print(f"Processing image {idx}: {image_path}")
    
    image = plt.imread(image_path)
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image / 255.0  # Normalize if needed
    
    if image.shape[-1] == 4:  # RGBA
        image_rgb = image[..., :3]  # Drop the alpha channel
    else:
        image_rgb = image  # Already in RGB or grayscale

    # If grayscale, convert to RGB
    if len(image_rgb.shape) == 2:  # Grayscale
        image_rgb = np.stack((image_rgb,) * 3, axis=-1)

    # Saturate the image
    image_rgb = saturate_image(image_rgb, saturation_factor=1.5)
    print(f"Image shape: {image_rgb.shape}, dtype: {image_rgb.dtype}")
    print(f"Image RGB range: {image_rgb.min():.2f} - {image_rgb.max():.2f}")

    # h, s, v = sample_hsv(image_rgb)  # Sample HSV value from the image

    target_rgb = sample_rgb(image_rgb)  # Sample RGB value from the image

    segments = hypersegmentation(
        image_rgb,
        method='felzenszwalb',
        scale=3,
        sigma=0.5,
        min_size=21
    )
    new_segments = filter_segments(
        image_rgb, segments,
        target_rgb=target_rgb,      # Use sampled RGB
        rgb_tolerance=0.20,         # Adjust as needed (0.0-1.0)
        min_area=21,
        max_eccentricity=.90,
        min_color_fraction=0.10,
    )

    print("Unique segment labels in new_segments:", np.unique(new_segments))
    print("Number of non-zero segments:", np.count_nonzero(new_segments))

    segments_dir = 'datasets/segments'
    if not os.path.exists(segments_dir):
        os.makedirs(segments_dir)

    segments_rgb = label2rgb(new_segments, image=image_rgb, bg_label=0)
    segments_file = os.path.join(segments_dir, f'segments_{idx}.png')
    plt.imsave(segments_file, segments_rgb)
    print(f"Saved segments image to {segments_file}\n")

    patches = extract_segment_patches(image, new_segments)
    save_patches(patches, segments_dir, idx)