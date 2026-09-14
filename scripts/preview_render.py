import numpy as np
from PIL import Image

# Let's inspect the dithered image and create an RGB visualization of the dark and light modes
dark_dither = np.array(Image.open("assets/dark_dither.png")) > 0
light_dither = np.array(Image.open("assets/light_dither.png")) > 0

# Dark palette: portrait #A78BFA, background #0A101F
dark_rgb = np.full((340, 300, 3), [10, 16, 31], dtype=np.uint8) # #0A101F
dark_rgb[dark_dither] = [167, 139, 250] # #A78BFA
Image.fromarray(dark_rgb).save("assets/dark_preview.png")

# Light palette: portrait #7C3AED, background #F8FAFC
light_rgb = np.full((340, 300, 3), [248, 250, 252], dtype=np.uint8) # #F8FAFC
light_rgb[light_dither] = [124, 58, 237] # #7C3AED
Image.fromarray(light_rgb).save("assets/light_preview.png")

print("Saved preview images.")
