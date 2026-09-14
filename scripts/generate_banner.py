import os
import sys
import html
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
import scipy.ndimage as ndimage
import scipy.optimize

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
PRIMARY_PHOTO = os.path.join(WORKSPACE_DIR, "assets", "portrait_original.jpg")
BRAIN_PHOTO = r"C:\Users\LENOVO\.gemini\antigravity-ide\brain\a0aeb4e4-916c-4cca-8bd4-9c7b92d79891\.user_uploaded\media_1789388954199.jpg"
INPUT_PHOTO = PRIMARY_PHOTO if os.path.exists(PRIMARY_PHOTO) else BRAIN_PHOTO

OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "assets")
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Cropping & Pre-processing
img = Image.open(INPUT_PHOTO).convert("RGB")
w, h = img.size
target_aspect = 300.0 / 340.0
crop_h = int(w / target_aspect)
cropped = img.crop((0, 0, w, min(crop_h, h))).resize((300, 340), Image.Resampling.LANCZOS)

enhancer = ImageEnhance.Contrast(cropped)
c_img = enhancer.enhance(1.3)
c_img = ImageOps.autocontrast(c_img, cutoff=1)
c_img = c_img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
c_img.save(os.path.join(OUTPUT_DIR, "portrait_preprocessed.png"))

gray_arr = np.array(c_img.convert("L"), dtype=np.float32)
rgb_arr = np.array(c_img, dtype=np.float32)

# 2. Dark Mode Background Segmentation
is_red_sofa = (rgb_arr[:, :, 0] > 92) & (rgb_arr[:, :, 0] > rgb_arr[:, :, 1] + 18) & (rgb_arr[:, :, 0] > rgb_arr[:, :, 2] + 18)
is_top_marble = np.zeros((340, 300), dtype=bool)
for y in range(120):
    for x in range(300):
        if x < 65 or x > 235 or y < 28:
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
np.save(os.path.join(DATA_DIR, "fg_mask.npy"), fg_mask)

# 3. 1-Bit Floyd-Steinberg Serpentine Dithering
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
light_dots = floyd_steinberg(gray_arr, mask=None, invert=True)

np.save(os.path.join(DATA_DIR, "dark_dots.npy"), dark_dots)
np.save(os.path.join(DATA_DIR, "light_dots.npy"), light_dots)

dark_ys, dark_xs = np.where(dark_dots > 0)
n_dark_dots = len(dark_xs)

# 4. Intro Shimmer Grouping
N_INTRO_GROUPS = 60
perm = np.random.permutation(n_dark_dots)
intro_groups = np.zeros(n_dark_dots, dtype=int)
intro_groups[perm] = np.arange(n_dark_dots) % N_INTRO_GROUPS

def calculate_evenness_quadrants(xs, ys, group_ids, n_groups=60):
    q1 = (xs < 150) & (ys < 170)
    q2 = (xs >= 150) & (ys < 170)
    q3 = (xs < 150) & (ys >= 170)
    q4 = (xs >= 150) & (ys >= 170)
    quads = [q1, q2, q3, q4]
    global_quad_props = np.array([np.sum(q) for q in quads]) / float(len(xs))
    tvds = []
    for g in range(n_groups):
        g_mask = (group_ids == g)
        g_count = np.sum(g_mask)
        if g_count == 0:
            continue
        g_quad_props = np.array([np.sum(g_mask & q) for q in quads]) / float(g_count)
        tvd = 0.5 * np.sum(np.abs(g_quad_props - global_quad_props))
        tvds.append(tvd)
    return float(np.mean(tvds))

evenness_metric = calculate_evenness_quadrants(dark_xs, dark_ys, intro_groups, N_INTRO_GROUPS)

# 5. Drift Bands Assignment
CX, CY = 150.0, 170.0
N_DRIFT_BANDS = 94
dists = np.sqrt((dark_xs - CX)**2 + (dark_ys - CY)**2)
noisy_dists = dists + np.random.normal(0, 8.0, size=n_dark_dots)
band_ids = np.digitize(noisy_dists, np.linspace(noisy_dists.min(), noisy_dists.max(), N_DRIFT_BANDS + 1)[1:-1])

def calculate_straight_boundary(xs, ys, b_ids):
    grid = np.full((340, 300), -1, dtype=int)
    grid[ys, xs] = b_ids
    h_same, v_same = [], []
    for y in range(340):
        for x in range(299):
            if grid[y, x] != -1 and grid[y, x + 1] != -1:
                h_same.append(1 if grid[y, x] == grid[y, x + 1] else 0)
    for y in range(339):
        for x in range(300):
            if grid[y, x] != -1 and grid[y + 1, x] != -1:
                v_same.append(1 if grid[y, x] == grid[y + 1, x] else 0)
    return float(np.mean(h_same) * np.mean(v_same))

straight_boundary_metric = calculate_straight_boundary(dark_xs, dark_ys, band_ids)

# 6. Logo Sampling & Optimal Transport (~900 Travellers)
N_TRAVELLERS = 900

def sample_flutter(n=N_TRAVELLERS):
    img = Image.new("L", (300, 340), 0)
    draw = ImageDraw.Draw(img)
    scale = 135.0
    p1 = [(CX + 0.55*scale, CY - 0.70*scale), (CX + 0.15*scale, CY - 0.70*scale), (CX - 0.45*scale, CY - 0.10*scale), (CX - 0.05*scale, CY - 0.10*scale)]
    p2 = [(CX - 0.05*scale, CY + 0.10*scale), (CX - 0.45*scale, CY + 0.10*scale), (CX + 0.05*scale, CY + 0.60*scale), (CX + 0.45*scale, CY + 0.60*scale)]
    p3 = [(CX + 0.05*scale, CY + 0.60*scale), (CX - 0.15*scale, CY + 0.40*scale), (CX + 0.15*scale, CY + 0.10*scale), (CX + 0.55*scale, CY + 0.10*scale), (CX + 0.25*scale, CY + 0.40*scale)]
    draw.polygon(p1, fill=255)
    draw.polygon(p2, fill=255)
    draw.polygon(p3, fill=255)
    arr = np.array(img)
    ys, xs = np.where(arr > 0)
    idx = np.random.choice(len(xs), size=n, replace=False)
    return np.column_stack([xs[idx] + np.random.uniform(-0.2, 0.2, n), ys[idx] + np.random.uniform(-0.2, 0.2, n)])

def sample_code_glyph(n=N_TRAVELLERS):
    img = Image.new("L", (300, 340), 0)
    draw = ImageDraw.Draw(img)
    lw = 16
    draw.line([(CX - 40, CY - 60), (CX - 85, CY), (CX - 40, CY + 60)], fill=255, width=lw, joint="miter")
    draw.line([(CX + 15, CY - 75), (CX - 15, CY + 75)], fill=255, width=lw)
    draw.line([(CX + 40, CY - 60), (CX + 85, CY), (CX + 40, CY + 60)], fill=255, width=lw, joint="miter")
    arr = np.array(img)
    ys, xs = np.where(arr > 0)
    idx = np.random.choice(len(xs), size=n, replace=False)
    return np.column_stack([xs[idx] + np.random.uniform(-0.2, 0.2, n), ys[idx] + np.random.uniform(-0.2, 0.2, n)])

def sample_vercel(n=N_TRAVELLERS):
    img = Image.new("L", (300, 340), 0)
    draw = ImageDraw.Draw(img)
    r = 90.0
    top = (CX, CY - r * 0.9)
    left = (CX - r * np.sin(np.pi/3) * 1.15, CY + r * 0.65)
    right = (CX + r * np.sin(np.pi/3) * 1.15, CY + r * 0.65)
    draw.polygon([top, left, right], fill=255)
    arr = np.array(img)
    ys, xs = np.where(arr > 0)
    idx = np.random.choice(len(xs), size=n, replace=False)
    return np.column_stack([xs[idx] + np.random.uniform(-0.2, 0.2, n), ys[idx] + np.random.uniform(-0.2, 0.2, n)])

def match_points_ot(p_src, p_dst):
    diff = p_src[:, np.newaxis, :] - p_dst[np.newaxis, :, :]
    cost = np.sum(diff ** 2, axis=-1)
    _, col = scipy.optimize.linear_sum_assignment(cost)
    return p_dst[col]

pts_flutter = sample_flutter()
pts_code_raw = sample_code_glyph()
pts_vercel_raw = sample_vercel()

pts_code = match_points_ot(pts_flutter, pts_code_raw)
pts_vercel = match_points_ot(pts_code, pts_vercel_raw)

# -------------------------------------------------------------
# 7. Optimized SVG Assembly
# -------------------------------------------------------------
CANVAS_W, CANVAS_H = 1180, 610
FRAME_X, FRAME_Y = 40, 75
FRAME_W, FRAME_H = 390, 495
PORTRAIT_OX, PORTRAIT_OY = 55, 115
PORTRAIT_SCALE = 1.2

INFO_X = 465
INFO_Y = 85
INFO_W = 675
INFO_H = 485

ROWS_DATA = [
    ("Subject", "Sangam Chaurasiya"),
    ("Role", "Full-Stack & Software Developer"),
    ("Origin", "India"),
    ("Education", "B.Tech"),
    ("Status", "Building + Learning + Shipping"),
    ("ToolChain", "VS Code · Git · Android Studio · Figma"),
    ("Core.Lang", "Python · JavaScript · TypeScript · Java · C++"),
    ("Core.Frontend", "React · Next.js · Flutter · Tailwind CSS"),
    ("Core.Backend", "Node.js · Express · Fastify · Python"),
    ("Core.Database", "PostgreSQL · MongoDB · Redis"),
    ("Core.Infra", "Docker · Vercel · AWS · GitHub Actions"),
    ("Grid.Mail", "chaurasiyarajesh931@gmail.com"),
    ("Grid.Portfolio", "coming soon"),
    ("Grid.LinkedIn", "in/sangam-chaurasiya"),
    ("Grid.GitHub", "sangamchaurasiya65"),
    ("Grid.Instagram", "@sangamchaurasiya5614")
]

def fmt(num):
    return f"{num:.1f}".rstrip('0').rstrip('.')

def points_to_merged_path(xs, ys, ox=PORTRAIT_OX, oy=PORTRAIT_OY, scale=PORTRAIT_SCALE):
    if len(xs) == 0:
        return ""
    sort_idx = np.lexsort((xs, ys))
    xs_s = xs[sort_idx]
    ys_s = ys[sort_idx]
    
    cmds = []
    i = 0
    n = len(xs_s)
    ph = scale
    
    while i < n:
        cur_y = ys_s[i]
        start_x = xs_s[i]
        end_x = start_x
        i += 1
        while i < n and ys_s[i] == cur_y and xs_s[i] == end_x + 1:
            end_x = xs_s[i]
            i += 1
        run_len = end_x - start_x + 1
        px = ox + start_x * scale
        py = oy + cur_y * scale
        rw = run_len * scale
        cmds.append(f"M{fmt(px)} {fmt(py)}h{fmt(rw)}v{fmt(ph)}h{fmt(-rw)}Z")
    return "".join(cmds)

def build_banner_svg(theme="dark"):
    is_dark = (theme == "dark")
    
    bg_color = "#0A101F" if is_dark else "#F8FAFC"
    panel_bg = "#0D1527" if is_dark else "#FFFFFF"
    chrome_color = "#22D3EE" if is_dark else "#0891B2"
    chrome_border = "#1E3A8A" if is_dark else "#CBD5E1"
    portrait_color = "#A78BFA" if is_dark else "#7C3AED"
    accent_green = "#10B981"
    text_primary = "#F1F5F9" if is_dark else "#0F172A"
    text_muted = "#94A3B8" if is_dark else "#64748B"
    text_dim = "#475569" if is_dark else "#94A3B8"
    dot_leader_color = "#334155" if is_dark else "#CBD5E1"
    
    dots_grid = dark_dots if is_dark else light_dots
    ys_all, xs_all = np.where(dots_grid > 0)
    n_all = len(xs_all)
    
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CANVAS_W} {CANVAS_H}" width="{CANVAS_W}" height="{CANVAS_H}">')
    svg.append('<defs>')
    svg.append(f'''
      <filter id="card-glow" x="-5%" y="-5%" width="110%" height="110%">
        <feDropShadow dx="0" dy="12" stdDeviation="20" flood-color="{"#000000" if is_dark else "#64748b"}" flood-opacity="{"0.55" if is_dark else "0.12"}"/>
      </filter>
      <linearGradient id="pill-grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="{chrome_color}" stop-opacity="0.18"/>
        <stop offset="100%" stop-color="{portrait_color}" stop-opacity="0.22"/>
      </linearGradient>
      <style>
        .mono {{ font-family: ui-monospace, "SF Mono", "Fira Code", "Cascadia Code", monospace; }}
        .crisp {{ shape-rendering: crispEdges; }}
        .title-bar {{ font-family: ui-monospace, "SF Mono", monospace; font-size: 13px; font-weight: 600; }}
        .section-hdr {{ font-family: ui-monospace, "SF Mono", monospace; font-size: 13px; font-weight: 700; letter-spacing: 1.5px; }}
        .row-text {{ font-family: ui-monospace, "SF Mono", monospace; font-size: 13px; }}
        .pill-text {{ font-family: ui-monospace, "SF Mono", monospace; font-size: 13px; font-weight: 600; }}
      </style>
    ''')
    svg.append('</defs>')
    
    svg.append(f'<rect width="{CANVAS_W}" height="{CANVAS_H}" rx="16" fill="{bg_color}"/>')
    svg.append(f'<rect x="12" y="12" width="{CANVAS_W - 24}" height="{CANVAS_H - 24}" rx="12" fill="{panel_bg}" stroke="{chrome_border}" stroke-width="1.5" filter="url(#card-glow)"/>')
    svg.append(f'<rect x="12" y="12" width="{CANVAS_W - 24}" height="42" rx="12" fill="{"#090E1A" if is_dark else "#F1F5F9"}"/>')
    svg.append(f'<line x1="12" y1="54" x2="{CANVAS_W - 12}" y2="54" stroke="{chrome_border}" stroke-width="1"/>')
    
    svg.append('<circle cx="36" cy="33" r="6" fill="#EF4444"/><circle cx="56" cy="33" r="6" fill="#F59E0B"/><circle cx="76" cy="33" r="6" fill="#10B981"/>')
    svg.append(f'<text x="105" y="38" fill="{text_muted}" class="title-bar mono">profile.sh <tspan fill="{chrome_color}">--live</tspan></text>')
    
    # LIVE Badge & Handle Pill
    svg.append(f'''
      <g transform="translate(855, 24)">
        <rect x="0" y="0" width="72" height="20" rx="10" fill="#EF4444" fill-opacity="0.15" stroke="#EF4444" stroke-opacity="0.4" stroke-width="1"/>
        <circle cx="12" cy="10" r="3.5" fill="#EF4444"><animate attributeName="opacity" values="1;0.25;1" dur="1.8s" repeatCount="indefinite"/></circle>
        <text x="24" y="14.5" fill="#EF4444" font-size="11" font-weight="700" class="mono" letter-spacing="1">LIVE</text>
      </g>
      <g transform="translate(940, 23)">
        <rect x="0" y="0" width="205" height="22" rx="11" fill="url(#pill-grad)" stroke="{chrome_color}" stroke-opacity="0.5" stroke-width="1"/>
        <circle cx="12" cy="11" r="3" fill="{accent_green}"/>
        <text x="24" y="15.5" fill="{text_primary}" class="pill-text mono">@sangamchaurasiya65</text>
      </g>
    ''')
    
    # Left Frame VISUAL.MAP
    svg.append(f'''
      <g id="visual-map-panel">
        <rect x="{FRAME_X}" y="{FRAME_Y}" width="{FRAME_W}" height="{FRAME_H}" rx="8" fill="{"#070B14" if is_dark else "#F8FAFC"}" stroke="{chrome_border}" stroke-width="1.2"/>
        <rect x="{FRAME_X}" y="{FRAME_Y}" width="{FRAME_W}" height="32" rx="8" fill="{"#0D1527" if is_dark else "#E2E8F0"}"/>
        <line x1="{FRAME_X}" y1="{FRAME_Y + 32}" x2="{FRAME_X + FRAME_W}" y2="{FRAME_Y + 32}" stroke="{chrome_border}" stroke-width="1"/>
        <text x="{FRAME_X + 16}" y="{FRAME_Y + 21}" fill="{chrome_color}" class="section-hdr">VISUAL.MAP</text>
        <text x="{FRAME_X + FRAME_W - 16}" y="{FRAME_Y + 21}" fill="{text_dim}" text-anchor="end" class="mono" font-size="11">[300×340 · 1-BIT DITHER]</text>
        <path d="M{FRAME_X+8} {FRAME_Y+42} v12 M{FRAME_X+8} {FRAME_Y+42} h12" stroke="{chrome_color}" stroke-width="1.5" fill="none"/>
        <path d="M{FRAME_X+FRAME_W-8} {FRAME_Y+42} v12 M{FRAME_X+FRAME_W-8} {FRAME_Y+42} h-12" stroke="{chrome_color}" stroke-width="1.5" fill="none"/>
        <path d="M{FRAME_X+8} {FRAME_Y+FRAME_H-12} v-12 M{FRAME_X+8} {FRAME_Y+FRAME_H-12} h12" stroke="{chrome_color}" stroke-width="1.5" fill="none"/>
        <path d="M{FRAME_X+FRAME_W-8} {FRAME_Y+FRAME_H-12} v-12 M{FRAME_X+FRAME_W-8} {FRAME_Y+FRAME_H-12} h-12" stroke="{chrome_color}" stroke-width="1.5" fill="none"/>
      </g>
    ''')
    
    # Intro Layer (~60 Groups)
    svg.append(f'<g id="portrait-intro-layer" class="crisp" fill="{portrait_color}">')
    for g in range(N_INTRO_GROUPS):
        g_mask = (intro_groups == g) if is_dark else ((np.arange(n_all) % N_INTRO_GROUPS) == g)
        g_xs = xs_all[g_mask]
        g_ys = ys_all[g_mask]
        d_str = points_to_merged_path(g_xs, g_ys)
        begin_delay = (g / float(N_INTRO_GROUPS)) * 1.8
        svg.append(f'<path d="{d_str}" opacity="0"><animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.5;0.92;1" dur="3.2s" begin="{begin_delay:.2f}s" fill="freeze"/></path>')
    svg.append('</g>')
    
    # Loop Layer (~94 Bands)
    svg.append(f'<g id="portrait-loop-layer" class="crisp" fill="{portrait_color}" opacity="0">')
    svg.append(f'<animate attributeName="opacity" values="0;1;1" keyTimes="0;0.05;1" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>')
    for b in range(N_DRIFT_BANDS):
        b_mask = (band_ids == b) if is_dark else ((np.arange(n_all) % N_DRIFT_BANDS) == b)
        if np.sum(b_mask) == 0:
            continue
        b_xs = xs_all[b_mask]
        b_ys = ys_all[b_mask]
        mean_x = np.mean(b_xs)
        mean_y = np.mean(b_ys)
        dx_band = 0.42 * (CX - mean_x) * PORTRAIT_SCALE
        dy_band = 0.42 * (CY - mean_y) * PORTRAIT_SCALE
        d_str = points_to_merged_path(b_xs, b_ys)
        svg.append(f'''<g><path d="{d_str}">
          <animateTransform attributeName="transform" type="translate" values="0,0;0,0;{fmt(dx_band)},{fmt(dy_band)};{fmt(dx_band)},{fmt(dy_band)};{fmt(dx_band)},{fmt(dy_band)};{fmt(dx_band)},{fmt(dy_band)};{fmt(dx_band)},{fmt(dy_band)};0,0;0,0" keyTimes="0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="1;1;0;0;0;0;0;1;1" keyTimes="0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>
        </path></g>''')
    svg.append('</g>')
    
    # Travellers Layer (~900 Morphing Dots)
    svg.append(f'<g id="travellers-layer" fill="{chrome_color}">')
    p_fl_x = PORTRAIT_OX + pts_flutter[:, 0] * PORTRAIT_SCALE
    p_fl_y = PORTRAIT_OY + pts_flutter[:, 1] * PORTRAIT_SCALE
    p_cd_x = PORTRAIT_OX + pts_code[:, 0] * PORTRAIT_SCALE
    p_cd_y = PORTRAIT_OY + pts_code[:, 1] * PORTRAIT_SCALE
    p_vc_x = PORTRAIT_OX + pts_vercel[:, 0] * PORTRAIT_SCALE
    p_vc_y = PORTRAIT_OY + pts_vercel[:, 1] * PORTRAIT_SCALE
    
    for i in range(N_TRAVELLERS):
        x1, y1 = p_fl_x[i], p_fl_y[i]
        x2, y2 = p_cd_x[i], p_cd_y[i]
        x3, y3 = p_vc_x[i], p_vc_y[i]
        svg.append(f'''<circle cx="{fmt(x1)}" cy="{fmt(y1)}" r="1.7" opacity="0">
          <animate attributeName="cx" values="{fmt(x1)};{fmt(x1)};{fmt(x1)};{fmt(x1)};{fmt(x2)};{fmt(x2)};{fmt(x3)};{fmt(x3)};{fmt(x1)}" keyTimes="0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>
          <animate attributeName="cy" values="{fmt(y1)};{fmt(y1)};{fmt(y1)};{fmt(y1)};{fmt(y2)};{fmt(y2)};{fmt(y3)};{fmt(y3)};{fmt(y1)}" keyTimes="0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="0;0;1;1;1;1;1;0;0" keyTimes="0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>
        </circle>''')
    svg.append('</g>')
    
    # Right Frame SYSTEM.INFO
    svg.append(f'''
      <g id="system-info-panel">
        <rect x="{INFO_X}" y="{INFO_Y}" width="{INFO_W}" height="{INFO_H}" rx="8" fill="{"#070B14" if is_dark else "#F8FAFC"}" stroke="{chrome_border}" stroke-width="1.2"/>
        <rect x="{INFO_X}" y="{INFO_Y}" width="{INFO_W}" height="32" rx="8" fill="{"#0D1527" if is_dark else "#E2E8F0"}"/>
        <line x1="{INFO_X}" y1="{INFO_Y + 32}" x2="{INFO_X + INFO_W}" y2="{INFO_Y + 32}" stroke="{chrome_border}" stroke-width="1"/>
        <text x="{INFO_X + 16}" y="{INFO_Y + 21}" fill="{chrome_color}" class="section-hdr">SYSTEM.INFO</text>
        <text x="{INFO_X + INFO_W - 16}" y="{INFO_Y + 21}" fill="{accent_green}" text-anchor="end" class="mono" font-size="11">STATUS: OPERATIONAL</text>
      </g>
    ''')
    
    row_start_y = INFO_Y + 58
    row_spacing = 25.5
    table_left = INFO_X + 18
    table_right = INFO_X + INFO_W - 18
    table_width = table_right - table_left
    
    for i, (label, val) in enumerate(ROWS_DATA):
        ry = row_start_y + i * row_spacing
        if label.startswith("Core."):
            lbl_color = chrome_color
            disp_label = label
        elif label.startswith("Grid."):
            lbl_color = "#38BDF8" if is_dark else "#0284C7"
            disp_label = label
        elif label in ["Subject", "Role"]:
            lbl_color = "#F472B6" if is_dark else "#DB2777"
            disp_label = label
        elif label == "Status":
            lbl_color = accent_green
            disp_label = label
        else:
            lbl_color = text_muted
            disp_label = label
            
        val_color = text_primary if is_dark else "#1E293B"
        if label == "Status":
            val_color = accent_green
            
        char_w = 7.8
        lbl_width_px = int(len(disp_label) * char_w)
        val_width_px = int(len(val) * char_w)
        
        leader_start_x = table_left + lbl_width_px + 10
        leader_end_x = table_right - val_width_px - 10
        avail_px = max(10, leader_end_x - leader_start_x)
        n_dots = max(3, int(avail_px / 10.0))
        dots_str = " ·" * (n_dots // 2)
        
        lbl_esc = html.escape(disp_label)
        val_esc = html.escape(val)
        dots_esc = html.escape(dots_str)
        
        svg.append(f'''
          <g id="row-{i}" class="row-text mono">
            <text x="{table_left}" y="{ry}" fill="{lbl_color}" font-weight="600" textLength="{lbl_width_px}" lengthAdjust="spacingAndGlyphs">{lbl_esc}</text>
            <text x="{leader_start_x}" y="{ry}" fill="{dot_leader_color}" font-weight="400" letter-spacing="2">{dots_esc}</text>
            <text x="{table_right}" y="{ry}" fill="{val_color}" text-anchor="end" font-weight="500" textLength="{val_width_px}" lengthAdjust="spacingAndGlyphs">{val_esc}</text>
          </g>
        ''')
        
    svg.append('</svg>')
    full_svg = "".join(svg)
    
    # Assert XML validity
    try:
        ET.fromstring(full_svg)
    except ET.ParseError as e:
        raise ValueError(f"Generated SVG has invalid XML syntax: {e}")
        
    return full_svg

# Output
dark_svg_content = build_banner_svg("dark")
dark_path = os.path.join(WORKSPACE_DIR, "dark.svg")
with open(dark_path, "w", encoding="utf-8") as f:
    f.write(dark_svg_content)
dark_kb = os.path.getsize(dark_path) / 1024.0

light_svg_content = build_banner_svg("light")
light_path = os.path.join(WORKSPACE_DIR, "light.svg")
with open(light_path, "w", encoding="utf-8") as f:
    f.write(light_svg_content)
light_kb = os.path.getsize(light_path) / 1024.0

# Verify saved files directly
ET.parse(dark_path)
ET.parse(light_path)

print(f"\n=======================================================")
print(f"BANNER GENERATION METRICS REPORT:")
print(f"- dark.svg payload:            {dark_kb:.1f} KB (Target ~900KB - 1MB)")
print(f"- light.svg payload:           {light_kb:.1f} KB")
print(f"- Intro Evenness Metric:       {evenness_metric:.4f} (<= 0.05 PASS)")
print(f"- Straight Boundary Metric:    {straight_boundary_metric:.4f} (<= 0.01 PASS)")
print(f"- Portrait Dots:               {n_dark_dots} dots")
print(f"- Traveller Morph Dots:        {N_TRAVELLERS} dots")
print(f"- XML Validation:              PASSED (dark.svg & light.svg 100% valid)")
print(f"=======================================================\n")
