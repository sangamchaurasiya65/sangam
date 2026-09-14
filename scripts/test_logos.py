import numpy as np
from PIL import Image, ImageDraw
import scipy.optimize

N_TRAVELLERS = 900
W, H = 300, 340
CX, CY = W / 2.0, H / 2.0

# 1. Flutter Logo Point Sampling
# Flutter logo geometry:
# Top wing: diamond/strip
# Bottom small chevron & lower wing
def get_flutter_points(n=N_TRAVELLERS):
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    
    # Flutter logo strokes centered at (CX, CY)
    # Scale factor
    scale = 140.0
    # Top chevron / bar:
    # Points in normalized [-1, 1] coords
    # Flutter has 3 parts:
    # 1. Top long strip: from top right down-left
    # 2. Middle small strip
    # 3. Bottom chevron pointing right
    # Let's draw high-res polygon
    p1 = [(CX + 0.55*scale, CY - 0.70*scale), (CX + 0.15*scale, CY - 0.70*scale), (CX - 0.45*scale, CY - 0.10*scale), (CX - 0.05*scale, CY - 0.10*scale)]
    # Middle small strip:
    p2 = [(CX - 0.05*scale, CY + 0.10*scale), (CX - 0.45*scale, CY + 0.10*scale), (CX + 0.05*scale, CY + 0.60*scale), (CX + 0.45*scale, CY + 0.60*scale)]
    # Bottom chevron:
    p3 = [(CX + 0.05*scale, CY + 0.60*scale), (CX - 0.15*scale, CY + 0.40*scale), (CX + 0.15*scale, CY + 0.10*scale), (CX + 0.55*scale, CY + 0.10*scale), (CX + 0.25*scale, CY + 0.40*scale)]
    
    draw.polygon(p1, fill=255)
    draw.polygon(p2, fill=255)
    draw.polygon(p3, fill=255)
    
    arr = np.array(img)
    ys, xs = np.where(arr > 0)
    if len(xs) < n:
        raise ValueError("Not enough pixels in Flutter raster")
    idx = np.random.choice(len(xs), size=n, replace=False)
    # Add slight sub-pixel jitter
    pts = np.column_stack([xs[idx] + np.random.uniform(-0.3, 0.3, n), ys[idx] + np.random.uniform(-0.3, 0.3, n)])
    return pts

# 2. Code Glyph </> Point Sampling
def get_code_glyph_points(n=N_TRAVELLERS):
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    
    # Left bracket <
    # Stroke width ~14px
    lw = 16
    draw.line([(CX - 40, CY - 60), (CX - 85, CY), (CX - 40, CY + 60)], fill=255, width=lw, joint="miter")
    # Slash /
    draw.line([(CX + 15, CY - 75), (CX - 15, CY + 75)], fill=255, width=lw)
    # Right bracket >
    draw.line([(CX + 40, CY - 60), (CX + 85, CY), (CX + 40, CY + 60)], fill=255, width=lw, joint="miter")
    
    arr = np.array(img)
    ys, xs = np.where(arr > 0)
    idx = np.random.choice(len(xs), size=n, replace=False)
    pts = np.column_stack([xs[idx] + np.random.uniform(-0.3, 0.3, n), ys[idx] + np.random.uniform(-0.3, 0.3, n)])
    return pts

# 3. Vercel Logo Point Sampling (Triangle)
def get_vercel_points(n=N_TRAVELLERS):
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    
    # Equilateral triangle centered at CX, CY
    r = 95.0
    top = (CX, CY - r * 0.9)
    left = (CX - r * np.sin(np.pi/3) * 1.15, CY + r * 0.65)
    right = (CX + r * np.sin(np.pi/3) * 1.15, CY + r * 0.65)
    
    draw.polygon([top, left, right], fill=255)
    
    arr = np.array(img)
    ys, xs = np.where(arr > 0)
    idx = np.random.choice(len(xs), size=n, replace=False)
    pts = np.column_stack([xs[idx] + np.random.uniform(-0.3, 0.3, n), ys[idx] + np.random.uniform(-0.3, 0.3, n)])
    return pts

# Optimal Transport Bipartite Matching (Hungarian / Linear Sum Assignment)
def match_points(pts_a, pts_b):
    # pts_a: (N, 2), pts_b: (N, 2)
    # Cost matrix: squared euclidean distances
    diff = pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]
    cost_matrix = np.sum(diff ** 2, axis=-1)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    # Return pts_b reordered to match pts_a
    return pts_b[col_ind]

np.random.seed(42)
pts_flutter = get_flutter_points()
pts_code_raw = get_code_glyph_points()
pts_vercel_raw = get_vercel_points()

# Optimal transport chaining:
pts_code = match_points(pts_flutter, pts_code_raw)
pts_vercel = match_points(pts_code, pts_vercel_raw)

print(f"Flutter points shape: {pts_flutter.shape}")
print(f"Code points shape: {pts_code.shape}")
print(f"Vercel points shape: {pts_vercel.shape}")

# Calculate average displacement across transitions
d1 = np.mean(np.linalg.norm(pts_flutter - pts_code, axis=1))
d2 = np.mean(np.linalg.norm(pts_code - pts_vercel, axis=1))
d3 = np.mean(np.linalg.norm(pts_vercel - pts_flutter, axis=1))
print(f"Average path displacement: Flutter->Code: {d1:.2f}px, Code->Vercel: {d2:.2f}px, Vercel->Flutter: {d3:.2f}px")
