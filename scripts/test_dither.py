import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import scipy.ndimage as ndimage

input_path = r"C:\Users\LENOVO\.gemini\antigravity-ide\brain\a0aeb4e4-916c-4cca-8bd4-9c7b92d79891\.user_uploaded\media_1789388954199.jpg"
img = Image.open(input_path).convert("RGB")
w, h = img.size
target_aspect = 300.0 / 340.0
crop_h = int(w / target_aspect)
crop_box = (0, 0, w, min(crop_h, h))
cropped = img.crop(crop_box).resize((300, 340), Image.Resampling.LANCZOS)

# 1. Apply image enhancements
# Contrast 1.3x only, autocontrast(cutoff=1), UnsharpMask(radius=3, percent=140)
enhancer = ImageEnhance.Contrast(cropped)
c_img = enhancer.enhance(1.3)
c_img = ImageOps.autocontrast(c_img, cutoff=1)
c_img = c_img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
c_img.save("assets/enhanced_300x340.png")

# Convert to grayscale / array
gray = np.array(c_img.convert("L"), dtype=np.float32)

# Background Segmentation for Dark Mode:
# In the photo, the background has red seats / gold-black backdrop.
# The subject is in dark blue blazer with glasses and dark hair.
# Let's inspect RGB color segmentation
rgb_arr = np.array(c_img, dtype=np.float32)

# Let's detect background vs foreground:
# Notice in dark mode, dots draw the lit subject (face, skin, highlights, tie/collar, glasses).
# For dark mode, background should be 0 (no dots / dark), and subject should have dots proportional to brightness/features.
# Let's inspect background color profile (e.g. top corners, red chair behind).
# Red chair: high R, lower G/B (R > 120 and R > G + 30 and R > B + 30).
# Gold/wall background: top area.
# Let's test morphological segmentation:
# Background seed points: top-left (0,0), top-right (299,0), middle left/right chair.
# Or thresholding:
is_red_chair = (rgb_arr[:, :, 0] > 100) & (rgb_arr[:, :, 0] > rgb_arr[:, :, 1] * 1.3) & (rgb_arr[:, :, 0] > rgb_arr[:, :, 2] * 1.3)
is_top_bg = np.zeros((340, 300), dtype=bool)
is_top_bg[:60, :] = (rgb_arr[:60, :, 0] > 60) & (rgb_arr[:60, :, 1] > 60) # wall / backdrop

bg_mask = is_red_chair | is_top_bg
# Foreground is roughly NOT bg_mask
fg_mask = ~bg_mask

# Binary closing and fill holes
fg_mask = ndimage.binary_closing(fg_mask, structure=np.ones((7, 7)))
fg_mask = ndimage.binary_fill_holes(fg_mask)

# Keep largest connected component
lbl, num = ndimage.label(fg_mask)
if num > 0:
    counts = np.bincount(lbl.flat)[1:]
    largest = counts.argmax() + 1
    fg_mask = (lbl == largest)

# Smooth mask
fg_mask = ndimage.binary_dilation(fg_mask, structure=np.ones((3, 3)))
fg_mask = ndimage.binary_fill_holes(fg_mask)

Image.fromarray((fg_mask * 255).astype(np.uint8)).save("assets/fg_mask.png")

# Floyd-Steinberg Dithering with Serpentine order
def floyd_steinberg(img_array, mask=None, invert=False):
    # img_array is 0..255 float
    h, w = img_array.shape
    arr = img_array.copy()
    if invert:
        arr = 255.0 - arr
    
    output = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(h):
        # Serpentine: even rows left-to-right, odd rows right-to-left
        if y % 2 == 0:
            x_range = range(w)
            direction = 1
        else:
            x_range = range(w - 1, -1, -1)
            direction = -1
            
        for x in x_range:
            if mask is not None and not mask[y, x]:
                # Masked out: force 0 and zero error bleed
                arr[y, x] = 0.0
                output[y, x] = 0
                continue
                
            old_val = arr[y, x]
            new_val = 255.0 if old_val >= 128.0 else 0.0
            output[y, x] = 1 if new_val == 255.0 else 0
            err = old_val - new_val
            
            # Distribute error
            # (y, x + dir): 7/16
            # (y + 1, x - dir): 3/16
            # (y + 1, x): 5/16
            # (y + 1, x + dir): 1/16
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

# Dark mode: dots draw the lit subject on dark panel
# For dark mode, lighter parts of the subject get dots.
# Let's adjust brightness/gamma of subject so face features, eyes, hair, clothes are beautifully articulated.
dark_dither = floyd_steinberg(gray, mask=fg_mask, invert=False)
Image.fromarray((dark_dither * 255).astype(np.uint8)).save("assets/dark_dither.png")
print(f"Dark mode dots count: {np.sum(dark_dither)}")

# Light mode: keep background; dots draw the dark parts (invert=True so darker regions become high intensity)
light_dither = floyd_steinberg(gray, mask=None, invert=True)
Image.fromarray((light_dither * 255).astype(np.uint8)).save("assets/light_dither.png")
print(f"Light mode dots count: {np.sum(light_dither)}")
