import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
import scipy.ndimage as ndimage
import scipy.optimize

WORKSPACE_DIR = r"c:\Users\LENOVO\OneDrive\Desktop\sangam"
INPUT_PHOTO = r"C:\Users\LENOVO\.gemini\antigravity-ide\brain\a0aeb4e4-916c-4cca-8bd4-9c7b92d79891\.user_uploaded\media_1789388954199.jpg"

np.random.seed(42)

# Load and crop photo
img = Image.open(INPUT_PHOTO).convert("RGB")
w, h = img.size
target_aspect = 300.0 / 340.0
crop_h = int(w / target_aspect)
cropped = img.crop((0, 0, w, min(crop_h, h))).resize((300, 340), Image.Resampling.LANCZOS)

# Filters: Contrast 1.3x, autocontrast(cutoff=1), UnsharpMask(radius=3, percent=140)
enhancer = ImageEnhance.Contrast(cropped)
c_img = enhancer.enhance(1.3)
c_img = ImageOps.autocontrast(c_img, cutoff=1)
c_img = c_img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))

gray_arr = np.array(c_img.convert("L"), dtype=np.float32)
rgb_arr = np.array(c_img, dtype=np.float32)

# Segmentation
is_red_sofa = (rgb_arr[:, :, 0] > 95) & (rgb_arr[:, :, 0] > rgb_arr[:, :, 1] + 20) & (rgb_arr[:, :, 0] > rgb_arr[:, :, 2] + 20)
is_top_marble = np.zeros((340, 300), dtype=bool)
for y in range(120):
    for x in range(300):
        if x < 65 or x > 235 or y < 30:
            is_top_marble[y, x] = True

bg_mask = is_red_sofa | is_top_marble
fg_mask = ~bg_mask
fg_mask = ndimage.binary_opening(fg_mask, structure=np.ones((3, 3)))
fg_mask = ndimage.binary_closing(fg_mask, structure=np.ones((9, 9)))
fg_mask = ndimage.binary_fill_holes(fg_mask)
lbl, num = ndimage.label(fg_mask)
if num > 0:
    counts = np.bincount(lbl.flat)[1:]
    largest = counts.argmax() + 1
    fg_mask = (lbl == largest)
fg_mask = ndimage.binary_fill_holes(fg_mask)

# Floyd-Steinberg Serpentine Dithering
def floyd_steinberg(img_array, mask=None, invert=False):
    h, w = img_array.shape
    arr = img_array.copy()
    if invert:
        arr = 255.0 - arr
    output = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        if y % 2 == 0:
            x_range = range(w)
            direction = 1
        else:
            x_range = range(w - 1, -1, -1)
            direction = -1
        for x in x_range:
            if mask is not None and not mask[y, x]:
                arr[y, x] = 0.0
                output[y, x] = 0
                continue
            old_val = arr[y, x]
            new_val = 255.0 if old_val >= 128.0 else 0.0
            output[y, x] = 1 if new_val == 255.0 else 0
            err = old_val - new_val
            if 0 <= x + direction < w:
                if mask is None or mask[y, x + direction]:
                    arr[y, x + direction] += err * (7.0 / 16.0)
            if y + 1 < h:
                if 0 <= x - direction < w:
                    if mask is None or mask[y + 1, x - direction]:
                        arr[y + 1, x - direction] += err * (3.0 / 16.0)
                if mask is None or mask[y + 1, x]:
                    arr[y + 1, x] += err * (5.0 / 16.0)
                if 0 <= x + direction < w:
                    if mask is None or mask[y + 1, x + direction]:
                        arr[y + 1, x + direction] += err * (1.0 / 16.0)
    return output

dark_dots = floyd_steinberg(gray_arr, mask=fg_mask, invert=False)
dark_ys, dark_xs = np.where(dark_dots > 0)
n_dots = len(dark_xs)
print(f"Total dark portrait dots: {n_dots}")

# 1. Stratified Evenness Assignment
# For each 6x6 spatial cell, distribute dots cyclically across 60 groups
N_INTRO_GROUPS = 60
intro_groups = np.zeros(n_dots, dtype=int)

cell_size = 6
cell_x = dark_xs // cell_size
cell_y = dark_ys // cell_size
cell_ids = cell_y * 1000 + cell_x
unique_cells = np.unique(cell_ids)

counter = 0
for cid in unique_cells:
    idx = np.where(cell_ids == cid)[0]
    # Random permutation within cell
    np.random.shuffle(idx)
    for i in idx:
        intro_groups[i] = counter % N_INTRO_GROUPS
        counter += 1

# Calculate Evenness Metric
n_bins_x, n_bins_y = 6, 6
bin_x = np.clip((dark_xs / 300.0 * n_bins_x).astype(int), 0, n_bins_x - 1)
bin_y = np.clip((dark_ys / 340.0 * n_bins_y).astype(int), 0, n_bins_y - 1)
bin_idx = bin_y * n_bins_x + bin_x
n_bins = n_bins_x * n_bins_y

global_hist = np.bincount(bin_idx, minlength=n_bins).astype(float)
global_dist = global_hist / np.sum(global_hist)

tvds = []
for g in range(N_INTRO_GROUPS):
    g_mask = (intro_groups == g)
    g_hist = np.bincount(bin_idx[g_mask], minlength=n_bins).astype(float)
    g_dist = g_hist / np.sum(g_hist)
    tvd = 0.5 * np.sum(np.abs(g_dist - global_dist))
    tvds.append(tvd)

evenness_metric = float(np.mean(tvds))
print(f"Stratified Evenness Metric: {evenness_metric:.4f} (Target <= 0.05)")

# 2. Straight Boundary Metric with Gaussian Noise
CX, CY = 150.0, 170.0
N_DRIFT_BANDS = 94
dists = np.sqrt((dark_xs - CX)**2 + (dark_ys - CY)**2)
noisy_dists = dists + np.random.normal(0, 6.0, size=n_dots)
band_ids = np.digitize(noisy_dists, np.linspace(noisy_dists.min(), noisy_dists.max(), N_DRIFT_BANDS + 1)[1:-1])

# Straight boundary test
grid = np.full((340, 300), -1, dtype=int)
grid[dark_ys, dark_xs] = band_ids
h_same, v_same = [], []
for y in range(340):
    for x in range(299):
        if grid[y, x] != -1 and grid[y, x + 1] != -1:
            h_same.append(1 if grid[y, x] == grid[y, x + 1] else 0)
for y in range(339):
    for x in range(300):
        if grid[y, x] != -1 and grid[y + 1, x] != -1:
            v_same.append(1 if grid[y, x] == grid[y + 1, x] else 0)
straight_metric = float(np.mean(h_same) * np.mean(v_same))
print(f"Straight Boundary Metric: {straight_metric:.4f} (Target <= 0.01)")
