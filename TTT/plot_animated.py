#!/usr/bin/env python3
"""
Animated version of the throughput plot.
Lines animate progressively, with the async update line visibly lifting at each update point.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
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

# --- Generate Base Throughput with Gradual Decay ---
def generate_base_throughput(n, base_level=380, noise=8):
    decay = np.linspace(0, -25, n)
    noise_signal = rng.normal(0, noise, n)
    low_freq = 15 * np.sin(2 * np.pi * np.arange(n) / (n / 3))
    return base_level + decay + noise_signal + low_freq

# --- Generate Async Weight Update Spikes ---
def add_async_weight_updates(y, update_times, spike_height=45, plateau_len=180, decay_rate=0.008):
    y_updated = y.copy()
    cumulative_boost = 0
    
    for i, t_idx in enumerate(update_times):
        if t_idx >= len(y):
            continue
        remaining = len(y) - t_idx
        spike_factor = rng.uniform(0.9, 1.1)
        actual_height = spike_height * spike_factor
        permanent_boost = 15 + i * 5
        cumulative_boost += permanent_boost
        boost = np.zeros(remaining)
        plateau_end = min(plateau_len, remaining)
        plateau_noise = rng.normal(0, 3, plateau_end)
        boost[:plateau_end] = actual_height + plateau_noise
        if remaining > plateau_len:
            decay_len = remaining - plateau_len
            temp_boost = actual_height - permanent_boost
            decay_curve = temp_boost * np.exp(-decay_rate * np.arange(decay_len))
            boost[plateau_len:] = permanent_boost + decay_curve + rng.normal(0, 2, decay_len)
        y_updated[t_idx:] += boost
    return y_updated

# --- Define Async Weight Update Times ---
update_indices = [450, 900, 1350]
update_times_min = [time_minutes[idx] for idx in update_indices]

# --- Generate Throughput Data ---
y_base = generate_base_throughput(n_points, base_level=340, noise=5)
y_with_updates = add_async_weight_updates(y_base.copy(), update_indices, spike_height=55, plateau_len=200, decay_rate=0.006)

# --- Smooth with Moving Average ---
win = 25
kernel = np.ones(win) / win
pad_len = win * 2
y_base_padded = np.concatenate([y_base, np.full(pad_len, y_base[-50:].mean())])
y_updates_padded = np.concatenate([y_with_updates, np.full(pad_len, y_with_updates[-50:].mean())])
y_base_smooth = np.convolve(y_base_padded, kernel, mode="same")[:n_points]
y_updates_smooth = np.convolve(y_updates_padded, kernel, mode="same")[:n_points]
y_base_smooth = np.clip(y_base_smooth, 280, 580)
y_updates_smooth = np.clip(y_updates_smooth, 280, 580)

# --- Animation Setup ---
# Downsample for smoother animation (every 10 points)
step = 10
n_frames = n_points // step + 1

fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

# Set background
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')

# Color palette
color_base = '#7f8c8d'
color_async = '#e74c3c'
color_accent = '#2ecc71'

# Initialize plot elements
line_base, = ax.plot([], [], color=color_base, linewidth=2.5, linestyle='--', 
                      alpha=0.8, label='Without Async Updates', zorder=2)
line_async, = ax.plot([], [], color=color_async, linewidth=3.0, 
                       label='With Async Weight Updates', zorder=3)
fill_collection = None

# Markers and annotations (will be added dynamically)
markers = []
annotations = []
vlines = []

# --- Styling ---
ax.set_xlabel('Time (minutes)', fontweight='bold', labelpad=10)
ax.set_ylabel('Throughput (tokens/s)', fontweight='bold', labelpad=10)
title = ax.set_title('Throughput Over Time with Async Weight Updates', 
                     fontweight='bold', pad=20, fontsize=15)
title.set_path_effects([path_effects.withSimplePatchShadow(offset=(1, -1), alpha=0.1)])
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)
ax.set_xlim(0, total_time_minutes + 0.5)
ax.set_ylim(290, 480)

legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
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
                                   edgecolor='#c0392b', alpha=0.95, linewidth=1.5),
                          color='#1a1a1a', fontweight='bold')

def init():
    line_base.set_data([], [])
    line_async.set_data([], [])
    stats_text_obj.set_text('')
    return [line_base, line_async, stats_text_obj]

def animate(frame):
    global fill_collection, markers, annotations, vlines
    
    # Current data point index
    idx = min(frame * step, n_points - 1)
    
    # Update line data
    x_data = time_minutes[:idx+1]
    y_base_data = y_base_smooth[:idx+1]
    y_async_data = y_updates_smooth[:idx+1]
    
    line_base.set_data(x_data, y_base_data)
    line_async.set_data(x_data, y_async_data)
    
    # Remove old fill
    if fill_collection is not None:
        fill_collection.remove()
    
    # Add new fill
    if len(x_data) > 1:
        fill_collection = ax.fill_between(x_data, 300, y_async_data, 
                                           alpha=0.15, color=color_async, zorder=1)
    
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
            
            # Vertical line
            vline = ax.axvline(x=t_min, color=color_accent, linestyle=':', 
                               linewidth=1.2, alpha=0.6, zorder=1)
            vlines.append(vline)
            
            # Diamond marker
            scatter = ax.scatter([t_min], [y_val], color=color_accent, s=80, 
                                  marker='D', zorder=5, edgecolors='white', linewidths=1.5)
            markers.append(scatter)
            
            # Annotation
            label_text = f'Async Update #{i+1}'
            ann = ax.annotate(label_text, (t_min, y_val), 
                              xytext=(0, 18), textcoords='offset points',
                              fontsize=9, ha='center', color='#1e8449',
                              fontweight='black',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                       edgecolor=color_accent, alpha=0.95, linewidth=1.2))
            annotations.append(ann)
    
    # Update stats
    if len(y_base_data) > 0:
        avg_base = np.mean(y_base_data)
        avg_async = np.mean(y_async_data)
        improvement = ((avg_async - avg_base) / avg_base) * 100
        stats_text = f'Baseline: {avg_base:.0f} t/s | Async: {avg_async:.0f} t/s | +{improvement:.1f}%'
        stats_text_obj.set_text(stats_text)
    
    return [line_base, line_async, stats_text_obj]

# Create animation
anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                frames=n_frames, interval=50, blit=False)

# Save as GIF
out_path = "./throughput_async_animated.gif"
print(f"Saving animation to: {out_path}")
print("This may take a minute...")

# Use pillow writer for GIF
anim.save(out_path, writer='pillow', fps=20)
plt.close()

print(f"Animation saved to: {out_path}")
