#!/usr/bin/env python3
"""
Plot Accept Length over Training Time
Demonstrates how accept length fluctuates but trends upward during training.
"""

import numpy as np
import matplotlib.pyplot as plt
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
total_time_minutes = 30  # Total duration in minutes
n_points = 1800  # Data points (1 per second)
time_seconds = np.linspace(0, total_time_minutes * 60, n_points)
time_minutes = time_seconds / 60

# --- Generate Base Accept Length with Upward Trend ---
def generate_base_accept_length(n, base_level=2.0, noise=0.08):
    """Generate baseline accept length with logarithmic upward trend."""
    # Logarithmic growth trend
    t_normalized = np.linspace(0, 1, n)
    growth = 3.0 * np.log1p(t_normalized * 5) / np.log1p(5)
    base_trend = base_level + growth
    
    # Add periodic fluctuations
    periodic1 = 0.15 * np.sin(2 * np.pi * np.arange(n) / (n / 8))
    periodic2 = 0.1 * np.sin(2 * np.pi * np.arange(n) / (n / 20) + 0.5)
    
    # Add random noise
    noise_signal = rng.normal(0, noise, n)
    
    return base_trend + periodic1 + periodic2 + noise_signal

# --- Add Async Weight Update Effects ---
def add_async_weight_updates(y, update_times, boost_height=0.3, fluctuation_len=120, decay_rate=0.015):
    """
    Add accept length boost after async weight updates.
    Each update causes:
    1. Short fluctuation/boost in accept length
    2. Gradual decay but maintains a portion of the gain
    """
    y_updated = y.copy()
    for t_idx in update_times:
        if t_idx >= len(y):
            continue
        remaining = len(y) - t_idx
        boost_factor = rng.uniform(0.9, 1.1)
        actual_height = boost_height * boost_factor
        
        # Create the boost curve
        boost = np.zeros(remaining)
        
        # Phase 1: Short fluctuation with boost
        fluct_end = min(fluctuation_len, remaining)
        # Quick rise then oscillation
        t_fluct = np.arange(fluct_end)
        fluct_signal = actual_height * np.exp(-decay_rate * t_fluct) * (1 + 0.3 * np.sin(t_fluct * 0.15))
        fluct_noise = rng.normal(0, 0.05, fluct_end)
        boost[:fluct_end] = fluct_signal + fluct_noise
        
        # Phase 2: Gradual decay but keep at least 50% of the boost
        if remaining > fluctuation_len:
            decay_len = remaining - fluctuation_len
            decay_curve = actual_height * np.exp(-0.005 * np.arange(decay_len))
            # Keep at least 50% of the boost permanently
            decay_curve = np.maximum(decay_curve, actual_height * 0.5)
            boost[fluctuation_len:] = decay_curve + rng.normal(0, 0.02, decay_len)
        
        y_updated[t_idx:] += boost
    return y_updated

# --- Define Async Weight Update Times (same as plot.py) ---
update_indices = [450, 900, 1350]  # At ~7.5, 15, 22.5 minutes
update_times_min = [time_minutes[idx] for idx in update_indices]

# --- Generate Accept Length Data ---
y_base = generate_base_accept_length(n_points, base_level=2.0, noise=0.06)
y_with_updates = add_async_weight_updates(y_base.copy(), update_indices, boost_height=0.35, fluctuation_len=150, decay_rate=0.012)

# --- Generate Baseline (without training) ---
baseline = 2.0 + rng.normal(0, 0.05, n_points)
baseline += 0.08 * np.sin(2 * np.pi * np.arange(n_points) / (n_points / 6))

# --- Smooth with Moving Average (fix edge effects) ---
win = 25
kernel = np.ones(win) / win

# Pad data to avoid edge effects at the end
pad_len = win * 2
y_base_padded = np.concatenate([y_base, np.full(pad_len, y_base[-50:].mean())])
y_updates_padded = np.concatenate([y_with_updates, np.full(pad_len, y_with_updates[-50:].mean())])
baseline_padded = np.concatenate([baseline, np.full(pad_len, baseline[-50:].mean())])

y_base_smooth = np.convolve(y_base_padded, kernel, mode="same")[:n_points]
y_updates_smooth = np.convolve(y_updates_padded, kernel, mode="same")[:n_points]
baseline_smooth = np.convolve(baseline_padded, kernel, mode="same")[:n_points]

# Clip to reasonable range
y_base_smooth = np.clip(y_base_smooth, 1.5, 7.0)
y_updates_smooth = np.clip(y_updates_smooth, 1.5, 7.0)
baseline_smooth = np.clip(baseline_smooth, 1.5, 7.0)

# --- Create Beautiful Plot ---
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

# Set background
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')

# Color palette - modern and visually appealing
color_baseline = '#7f8c8d'      # Gray for baseline
color_training = '#3498db'      # Blue for training progress
color_accent = '#2ecc71'        # Green for update markers (same as plot.py)

# Plot baseline - dashed for comparison
ax.plot(time_minutes, baseline_smooth, 
        color=color_baseline, linewidth=2.5, linestyle='--', 
        alpha=0.8, label='Without Training', zorder=2)

# Plot training progress with async updates - solid, prominent
ax.plot(time_minutes, y_updates_smooth, 
        color=color_training, linewidth=3.0, 
        label='With Async Weight Updates', zorder=3)

# Add gradient fill under the training curve
ax.fill_between(time_minutes, 1.8, y_updates_smooth, 
                alpha=0.15, color=color_training, zorder=1)

# --- Mark Async Weight Update Events ---
for i, (t_min, t_idx) in enumerate(zip(update_times_min, update_indices)):
    y_val = y_updates_smooth[t_idx]
    
    # Vertical line at update point
    ax.axvline(x=t_min, color=color_accent, linestyle=':', 
               linewidth=1.2, alpha=0.6, zorder=1)
    
    # Diamond marker at the update point
    ax.scatter([t_min], [y_val], color=color_accent, s=80, 
               marker='D', zorder=5, edgecolors='white', linewidths=1.5)
    
    # Add label for updates
    label_text = f'Async Update #{i+1}'
    y_offset = 18
    ax.annotate(label_text, (t_min, y_val), 
                xytext=(0, y_offset), textcoords='offset points',
                fontsize=9, ha='center', color='#1e8449',
                fontweight='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor=color_accent, alpha=0.95, linewidth=1.2))

# --- Styling ---
ax.set_xlabel('Time (minutes)', fontweight='bold', labelpad=10)
ax.set_ylabel('Accept Length', fontweight='bold', labelpad=10)

# Title with shadow effect
title = ax.set_title('Accept Length Over Time with Async Weight Updates', 
                     fontweight='bold', pad=20, fontsize=15)
title.set_path_effects([path_effects.withSimplePatchShadow(offset=(1, -1), alpha=0.1)])

# Grid styling
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Axis limits
ax.set_xlim(0, total_time_minutes + 0.5)
ax.set_ylim(1.8, max(y_updates_smooth) * 1.1)

# Legend with custom styling
legend = ax.legend(loc='lower right', frameon=True, fancybox=True, 
                   framealpha=0.95, shadow=True, borderpad=0.6,
                   prop={'weight': 'bold', 'size': 9})
legend.get_frame().set_edgecolor('#888888')
legend.get_frame().set_linewidth(1.2)

# Add statistics annotations
avg_baseline = np.mean(baseline_smooth)
avg_training = np.mean(y_updates_smooth)
final_training = np.mean(y_updates_smooth[-100:])
improvement = ((final_training - avg_baseline) / avg_baseline) * 100

stats_text = f'Initial: {avg_baseline:.2f} | Final: {final_training:.2f} | +{improvement:.1f}%'
ax.text(0.98, 0.03, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                 edgecolor='#2980b9', alpha=0.95, linewidth=1.5),
        color='#1a1a1a', fontweight='bold')

# Spine styling
for spine in ax.spines.values():
    spine.set_color('#cccccc')
    spine.set_linewidth(1)

# --- Save ---
out_path = "./accept_length_async_weight_update.png"
plt.tight_layout()
plt.savefig(out_path, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"Figure saved to: {out_path}")
