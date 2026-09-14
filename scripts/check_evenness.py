import numpy as np

# Load dark dots
dark_dots = np.load("data/dark_dots.npy")
dark_ys, dark_xs = np.where(dark_dots > 0)
n_dots = len(dark_xs)
N_INTRO_GROUPS = 60

# Interleaved cyclic assignment across sorted coordinates (Hilbert curve / scanline)
# Sort by (y * 300 + x)
sort_order = np.argsort(dark_ys * 300 + dark_xs)
intro_groups = np.zeros(n_dots, dtype=int)
intro_groups[sort_order] = np.arange(n_dots) % N_INTRO_GROUPS

# Evenness calculation over 6x6 spatial macro-regions
n_bins = 6
bin_x = np.clip((dark_xs / 300.0 * n_bins).astype(int), 0, n_bins - 1)
bin_y = np.clip((dark_ys / 340.0 * n_bins).astype(int), 0, n_bins - 1)
bin_idx = bin_y * n_bins + bin_x
total_bins = n_bins * n_bins

global_hist = np.bincount(bin_idx, minlength=total_bins).astype(float)
global_dist = global_hist / np.sum(global_hist)

tvds = []
for g in range(N_INTRO_GROUPS):
    g_mask = (intro_groups == g)
    g_hist = np.bincount(bin_idx[g_mask], minlength=total_bins).astype(float)
    g_dist = g_hist / np.sum(g_hist)
    tvd = 0.5 * np.sum(np.abs(g_dist - global_dist))
    tvds.append(tvd)

print(f"Cyclic Scanline Evenness Metric: {np.mean(tvds):.4f}")
