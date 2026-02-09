#!/usr/bin/env python3
"""
Animated version of the Accept Length plot.
Lines animate progressively, with the async update line visibly lifting at each update point.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects

# --- Style Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'font.weight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'text.color': '#1a1a1a',
    'axes.labelcolor': '#1a1a1a',
    'xtick.color': '#1a1a1a',
    'ytick.color': '#1a1a1a',
})

# --- Parameters ---
rng = np.random.default_rng(42)
total_time_minutes = 30
n_points = 1800
time_seconds = np.linspace(0, total_time_minutes * 60, n_points)
time_minutes = time_seconds / 60

# --- Generate Base Accept Length with Upward Trend ---
def generate_base_accept_length(n, base_level=2.0, noise=0.08):
    """Generate baseline accept length with logarithmic upward trend."""
    t_normalized = np.linspace(0, 1, n)
    growth = 3.0 * np.log1p(t_normalized * 5) / np.log1p(5)
    base_trend = base_level + growth
    periodic1 = 0.15 * np.sin(2 * np.pi * np.arange(n) / (n / 8))
    periodic2 = 0.1 * np.sin(2 * np.pi * np.arange(n) / (n / 20) + 0.5)
    noise_signal = rng.normal(0, noise, n)
    return base_trend + periodic1 + periodic2 + noise_signal

# --- Add Async Weight Update Effects ---
def add_async_weight_updates(y, update_times, boost_height=0.3, fluctuation_len=120, decay_rate=0.015):
    y_updated = y.copy()
    for t_idx in update_times:
        if t_idx >= len(y):
            continue
        remaining = len(y) - t_idx
        boost_factor = rng.uniform(0.9, 1.1)
        actual_height = boost_height * boost_factor
        boost = np.zeros(remaining)
        fluct_end = min(fluctuation_len, remaining)
        t_fluct = np.arange(fluct_end)
        fluct_signal = actual_height * np.exp(-decay_rate * t_fluct) * (1 + 0.3 * np.sin(t_fluct * 0.15))
        fluct_noise = rng.normal(0, 0.05, fluct_end)
        boost[:fluct_end] = fluct_signal + fluct_noise
        if remaining > fluctuation_len:
            decay_len = remaining - fluctuation_len
            decay_curve = actual_height * np.exp(-0.005 * np.arange(decay_len))
            decay_curve = np.maximum(decay_curve, actual_height * 0.5)
            boost[fluctuation_len:] = decay_curve + rng.normal(0, 0.02, decay_len)
        y_updated[t_idx:] += boost
    return y_updated

# --- Define Async Weight Update Times ---
update_indices = [450, 900, 1350]
update_times_min = [time_minutes[idx] for idx in update_indices]

# --- Generate Accept Length Data ---
y_base = generate_base_accept_length(n_points, base_level=2.0, noise=0.06)
y_with_updates = add_async_weight_updates(y_base.copy(), update_indices, boost_height=0.35, fluctuation_len=150, decay_rate=0.012)

# --- Generate Baseline (without training) ---
baseline = 2.0 + rng.normal(0, 0.05, n_points)
baseline += 0.08 * np.sin(2 * np.pi * np.arange(n_points) / (n_points / 6))

# --- Smooth with Moving Average ---
win = 25
kernel = np.ones(win) / win
pad_len = win * 2
y_base_padded = np.concatenate([y_base, np.full(pad_len, y_base[-50:].mean())])
y_updates_padded = np.concatenate([y_with_updates, np.full(pad_len, y_with_updates[-50:].mean())])
baseline_padded = np.concatenate([baseline, np.full(pad_len, baseline[-50:].mean())])

y_base_smooth = np.convolve(y_base_padded, kernel, mode="same")[:n_points]
y_updates_smooth = np.convolve(y_updates_padded, kernel, mode="same")[:n_points]
baseline_smooth = np.convolve(baseline_padded, kernel, mode="same")[:n_points]

y_base_smooth = np.clip(y_base_smooth, 1.5, 7.0)
y_updates_smooth = np.clip(y_updates_smooth, 1.5, 7.0)
baseline_smooth = np.clip(baseline_smooth, 1.5, 7.0)

# Calculate y limits
y_max = max(y_updates_smooth) * 1.1

# --- Animation Setup ---
step = 10
n_frames = n_points // step + 1

fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

# Set background
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')

# Color palette
color_baseline = '#7f8c8d'
color_training = '#3498db'
color_accent = '#2ecc71'

# Initialize plot elements
line_baseline, = ax.plot([], [], color=color_baseline, linewidth=2.5, linestyle='--', 
                          alpha=0.8, label='Without Training', zorder=2)
line_training, = ax.plot([], [], color=color_training, linewidth=3.0, 
                          label='With Async Weight Updates', zorder=3)
fill_collection = None

# Markers and annotations
markers = []
annotations = []
vlines = []

# --- Styling ---
ax.set_xlabel('Time (minutes)', fontweight='bold', labelpad=10)
ax.set_ylabel('Accept Length', fontweight='bold', labelpad=10)
title = ax.set_title('Accept Length Over Time with Async Weight Updates', 
                     fontweight='bold', pad=20, fontsize=15)
title.set_path_effects([path_effects.withSimplePatchShadow(offset=(1, -1), alpha=0.1)])
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)
ax.set_xlim(0, total_time_minutes + 0.5)
ax.set_ylim(1.8, y_max)

legend = ax.legend(loc='lower right', frameon=True, fancybox=True, 
                   framealpha=0.95, shadow=True, borderpad=0.6,
                   prop={'weight': 'bold', 'size': 9})
legend.get_frame().set_edgecolor('#888888')
legend.get_frame().set_linewidth(1.2)

for spine in ax.spines.values():
    spine.set_color('#cccccc')
    spine.set_linewidth(1)

# Stats text placeholder
stats_text_obj = ax.text(0.98, 0.03, '', transform=ax.transAxes, fontsize=9,
                          verticalalignment='bottom', horizontalalignment='right',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='#2980b9', alpha=0.95, linewidth=1.5),
                          color='#1a1a1a', fontweight='bold')

def init():
    line_baseline.set_data([], [])
    line_training.set_data([], [])
    stats_text_obj.set_text('')
    return [line_baseline, line_training, stats_text_obj]

def animate(frame):
    global fill_collection, markers, annotations, vlines
    
    idx = min(frame * step, n_points - 1)
    
    x_data = time_minutes[:idx+1]
    y_baseline_data = baseline_smooth[:idx+1]
    y_training_data = y_updates_smooth[:idx+1]
    
    line_baseline.set_data(x_data, y_baseline_data)
    line_training.set_data(x_data, y_training_data)
    
    # Remove old fill
    if fill_collection is not None:
        fill_collection.remove()
    
    # Add new fill
    if len(x_data) > 1:
        fill_collection = ax.fill_between(x_data, 1.8, y_training_data, 
                                           alpha=0.15, color=color_training, zorder=1)
    
    # Clear old markers and annotations
    for m in markers:
        m.remove()
    for a in annotations:
        a.remove()
    for v in vlines:
        v.remove()
    markers = []
    annotations = []
    vlines = []
    
    # Add markers for updates that have occurred
    for i, (t_min, t_idx) in enumerate(zip(update_times_min, update_indices)):
        if t_idx <= idx:
            y_val = y_updates_smooth[t_idx]
            
            vline = ax.axvline(x=t_min, color=color_accent, linestyle=':', 
                               linewidth=1.2, alpha=0.6, zorder=1)
            vlines.append(vline)
            
            scatter = ax.scatter([t_min], [y_val], color=color_accent, s=80, 
                                  marker='D', zorder=5, edgecolors='white', linewidths=1.5)
            markers.append(scatter)
            
            label_text = f'Async Update #{i+1}'
            ann = ax.annotate(label_text, (t_min, y_val), 
                              xytext=(0, 18), textcoords='offset points',
                              fontsize=9, ha='center', color='#1e8449',
                              fontweight='black',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                       edgecolor=color_accent, alpha=0.95, linewidth=1.2))
            annotations.append(ann)
    
    # Update stats
    if len(y_baseline_data) > 0:
        avg_baseline = np.mean(y_baseline_data)
        final_training = y_training_data[-1] if len(y_training_data) > 0 else avg_baseline
        improvement = ((final_training - avg_baseline) / avg_baseline) * 100
        stats_text = f'Initial: {avg_baseline:.2f} | Current: {final_training:.2f} | +{improvement:.1f}%'
        stats_text_obj.set_text(stats_text)
    
    return [line_baseline, line_training, stats_text_obj]

# Create animation
anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                frames=n_frames, interval=50, blit=False)

# Save as GIF
out_path = "./accept_length_async_animated.gif"
print(f"Saving animation to: {out_path}")
print("This may take a minute...")

anim.save(out_path, writer='pillow', fps=20)
plt.close()

print(f"Animation saved to: {out_path}")
