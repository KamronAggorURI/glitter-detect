import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, mark_boundaries
from skimage.util import img_as_float
from skimage.color import rgb2gray, label2rgb, rgb2hsv
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects, binary_opening, disk
import os
import glob

def hypersegmentation(img, method='felzenszwalb', **kwargs):
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
    target_hsv=None, hsv_tolerance=(0.05, 0.3, 0.3),  # (H, S, V) tolerances in [0,1]
    max_eccentricity=0.60, min_color_fraction=0.5
):
    labels = label(segments)
    props = regionprops(labels)
    mask = np.zeros_like(segments, dtype=bool)
    gray = rgb2gray(image)
    hsv = rgb2hsv(image)

    valid_indices = []
    for prop in props:
        area = prop.area
        y0, x0, y1, x1 = prop.bbox
        aspect_ratio = (x1 - x0 + 1) / (y1 - y0 + 1)
        segment_mask = (labels == prop.label)
        mean_brightness = gray[segment_mask].mean()
        eccentricity = prop.eccentricity

        # Color filtering
        if target_hsv is not None:
            h, s, v = hsv[:, :, 0][segment_mask], hsv[:, :, 1][segment_mask], hsv[:, :, 2][segment_mask]
            h0, s0, v0 = target_hsv
            dh, ds, dv = hsv_tolerance

            # Allow matches based on proximity to target even if just a part of the segment
            hue_dist = np.minimum(np.abs(h - h0), 1 - np.abs(h - h0))
            color_pixels = (
            (hue_dist <= dh) &
            (s >= s0 - ds) &
            (v >= v0 - dv)
            )
            # Calculate the fraction of color pixels in the segment
            color_fraction = color_pixels.sum() / h.size

            # Handle hue wraparound
            hue_dist = np.minimum(np.abs(h - h0), 1 - np.abs(h - h0))
            color_pixels = (
                (hue_dist <= dh) &
                (np.abs(s - s0) <= ds) &
                (np.abs(v - v0) <= dv)
            )
            color_fraction = color_pixels.sum() / h.size
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

    h, s, v = sample_hsv(image_rgb)  # Sample HSV value from the image

    segments = hypersegmentation(
        image_rgb,
        method='felzenszwalb', # Change to 'slic' if needed
        scale=3,  # Scale parameter for Felzenszwalb
        sigma=0.5, # Sigma for Gaussian smoothing
        min_size=30 # Minimum segment size
    )
    target_hsv = hsv_deg_to_unit(h, s, v)
    new_segments = filter_segments(
        image_rgb, segments,
        target_hsv=target_hsv,  # Orange hue range
        hsv_tolerance=(0.15, 0.5, 0.5),  # Tolerances for H, S, V
        min_area=5,  # Minimum area of segments
        max_eccentricity=.95,  # Maximum eccentricity of segments
        min_color_fraction=0.05,  # Minimum fraction of color pixels
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