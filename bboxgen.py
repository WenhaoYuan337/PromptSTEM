import os
import cv2
import csv
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter, label, find_objects
from skimage.exposure import rescale_intensity

image_path = 'result_zeolite/69/vis/epoch_24_train_cam.png'
output_base_dir = 'bbox_compare_methods'
os.makedirs(output_base_dir, exist_ok=True)

def dog_enhance(gray, sigma1=1, sigma2=4):
    blur1 = gaussian_filter(gray, sigma=sigma1)
    blur2 = gaussian_filter(gray, sigma=sigma2)
    dog = blur1 - blur2
    return rescale_intensity(dog, out_range=(0, 1))

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_uint8 = (gray * 255).astype(np.uint8)
    return clahe.apply(gray_uint8) / 255.0

def sigmoid_contrast(gray, gain=10, cutoff=0.5):
    return 1 / (1 + np.exp(-gain * (gray - cutoff)))

def adaptive_gamma(gray, gamma_dark=2.0, gamma_bright=0.5):
    mean_val = np.mean(gray)
    out = np.where(
        gray < mean_val,
        np.power(gray, gamma_dark),
        np.power(gray, gamma_bright)
    )
    return rescale_intensity(out, out_range=(0, 1))

def multi_threshold_filter(gray, thresholds=[95, 97, 99], min_area=15, expand_ratio=1.2, std_thresh=0.07):
    total_mask = np.zeros_like(gray, dtype=bool)
    for t in thresholds:
        binary = gray > np.percentile(gray, t)
        total_mask |= binary

    labeled, _ = label(total_mask)
    objects = find_objects(labeled)
    expanded_bboxes = []
    h, w = gray.shape

    for slc in objects:
        y_min, x_min = slc[0].start, slc[1].start
        y_max, x_max = slc[0].stop, slc[1].stop
        area = (x_max - x_min) * (y_max - y_min)
        if area < min_area:
            continue

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        bw = (x_max - x_min) * expand_ratio
        bh = (y_max - y_min) * expand_ratio

        x_min_exp = int(max(cx - bw / 2, 0))
        x_max_exp = int(min(cx + bw / 2, w))
        y_min_exp = int(max(cy - bh / 2, 0))
        y_max_exp = int(min(cy + bh / 2, h))

        region = gray[y_min_exp:y_max_exp, x_min_exp:x_max_exp]
        std = np.std(region)
        if std >= std_thresh:
            expanded_bboxes.append((x_min_exp, y_min_exp, x_max_exp, y_max_exp))

    return expanded_bboxes, total_mask

def draw_bboxes(image, bboxes, save_path):
    img_vis = (image.copy() * 255).astype(np.uint8)
    for (x_min, y_min, x_max, y_max) in bboxes:
        cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    cv2.imwrite(save_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

def save_bboxes_to_csv(bboxes, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x_min', 'y_min', 'x_max', 'y_max'])
        writer.writerows(bboxes)

image = imread(image_path)
gray = rgb2gray(image)
dog = dog_enhance(gray)


methods = {
    'clahe': apply_clahe(dog),
    'sigmoid': sigmoid_contrast(dog, gain=10, cutoff=np.mean(dog)),
    'gamma': adaptive_gamma(dog)
}

methods['dog_only'] = dog

for method_name, enhanced_img in methods.items():
    method_dir = os.path.join(output_base_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    imsave(os.path.join(method_dir, f"{method_name}_enhanced.png"), (enhanced_img * 255).astype(np.uint8))

    bboxes, mask = multi_threshold_filter(enhanced_img)

    imsave(os.path.join(method_dir, f"{method_name}_mask.png"), (mask.astype(np.uint8) * 255))

    draw_bboxes(enhanced_img, bboxes, os.path.join(method_dir, f"{method_name}_bbox.png"))
    save_bboxes_to_csv(bboxes, os.path.join(method_dir, f"{method_name}_bbox.csv"))

    print(f"[{method_name}] counts: {len(bboxes)}")

