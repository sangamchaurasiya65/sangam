import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import scipy.ndimage as ndimage

input_path = r"C:\Users\LENOVO\.gemini\antigravity-ide\brain\a0aeb4e4-916c-4cca-8bd4-9c7b92d79891\.user_uploaded\media_1789388954199.jpg"
img = Image.open(input_path).convert("RGB")
w, h = img.size
target_aspect = 300.0 / 340.0
crop_h = int(w / target_aspect)
cropped = img.crop((0, 0, w, crop_h)).resize((300, 340), Image.Resampling.LANCZOS)
rgb = np.array(cropped, dtype=np.float32)

# Background seed points:
# (0,0) top-left corner, (299,0) top-right corner, (0, 160) left sofa, (299, 160) right sofa
# Measure color distance or specific classifiers:
is_red_sofa = (rgb[:, :, 0] > 90) & (rgb[:, :, 0] > rgb[:, :, 1] + 25) & (rgb[:, :, 0] > rgb[:, :, 2] + 25)

# Marble wall: dark with gold streaks at top
is_top_marble = np.zeros((340, 300), dtype=bool)
# Head is roughly at x in [70, 230], y in [30, 200]
for y in range(120):
    for x in range(300):
        # If outside head region or matching wall texture
        if (x < 70 or x > 230 or y < 35):
            # Check if not hair
            is_top_marble[y, x] = True

bg_mask = is_red_sofa | is_top_marble

# Refine foreground mask
fg_mask = ~bg_mask

# Remove stray noise
fg_mask = ndimage.binary_opening(fg_mask, structure=np.ones((3, 3)))
# Binary closing to fill subject interior
fg_mask = ndimage.binary_closing(fg_mask, structure=np.ones((9, 9)))
# Fill holes
fg_mask = ndimage.binary_fill_holes(fg_mask)

# Keep largest connected component (the person)
lbl, num = ndimage.label(fg_mask)
if num > 0:
    counts = np.bincount(lbl.flat)[1:]
    largest = counts.argmax() + 1
    fg_mask = (lbl == largest)

# Smooth edges
fg_mask = ndimage.binary_fill_holes(fg_mask)

# Save mask
Image.fromarray((fg_mask * 255).astype(np.uint8)).save("assets/refined_fg_mask.png")
print(f"Foreground mask pixels: {np.sum(fg_mask)} / {300*340} ({np.sum(fg_mask)/(300*340)*100:.1f}%)")
