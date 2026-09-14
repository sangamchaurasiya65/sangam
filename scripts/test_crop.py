import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import scipy.ndimage as ndimage

input_path = r"C:\Users\LENOVO\.gemini\antigravity-ide\brain\a0aeb4e4-916c-4cca-8bd4-9c7b92d79891\.user_uploaded\media_1789388954199.jpg"
os.makedirs("assets", exist_ok=True)
os.makedirs("scripts", exist_ok=True)

img = Image.open(input_path).convert("RGB")
print(f"Original image size: {img.size}")

# The photo is portrait orientation. Let's crop head + shoulders.
# Target aspect ratio is 300 x 340 (w:h = 300:340 = 0.88235)
w, h = img.size
target_aspect = 300.0 / 340.0

# Head and shoulders: crop from top y=0 to y = int(w / target_aspect)
crop_h = int(w / target_aspect)
if crop_h > h:
    crop_w = int(h * target_aspect)
    crop_box = ((w - crop_w) // 2, 0, (w + crop_w) // 2, h)
else:
    # Keep head and shoulders with good breathing room
    crop_box = (0, 0, w, crop_h)

cropped = img.crop(crop_box)
cropped.save("assets/cropped_ref.jpg", quality=95)
print(f"Cropped size: {cropped.size}")

# Resize to target grid 300x340
grid_img = cropped.resize((300, 340), Image.Resampling.LANCZOS)
grid_img.save("assets/grid_300x340.png")
print("Grid image saved.")
