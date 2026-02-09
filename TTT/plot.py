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

# --- Generate Base Throughput with Gradual Decay ---
def generate_base_throughput(n, base_level=380, noise=8):
    """Generate baseline throughput with slight natural decay."""
    decay = np.linspace(0, -25, n)
    noise_signal = rng.normal(0, noise, n)
    # Add some low-frequency variation
    low_freq = 15 * np.sin(2 * np.pi * np.arange(n) / (n / 3))
    return base_level + decay + noise_signal + low_freq

# --- Generate Async Weight Update Spikes ---
def add_async_weight_updates(y, update_times, spike_height=45, plateau_len=180, decay_rate=0.008):
    """
    Add throughput spikes representing async weight updates.
    Each update causes:
    1. Sharp rise to spike_height
    2. Stable plateau with small fluctuations
    3. Gradual decay (but maintains a cumulative boost)
    """
    y_updated = y.copy()
    cumulative_boost = 0  # Each update adds permanent boost
    
    for i, t_idx in enumerate(update_times):
        if t_idx >= len(y):
            continue
        remaining = len(y) - t_idx
        spike_factor = rng.uniform(0.9, 1.1)
        actual_height = spike_height * spike_factor
        
        # Each update adds a permanent baseline boost
        permanent_boost = 15 + i * 5  # Increasing permanent boost
        cumulative_boost += permanent_boost
        
        # Create the boost curve: plateau then gentle decay to permanent level
        boost = np.zeros(remaining)
        
        # Phase 1: Plateau with small fluctuations
        plateau_end = min(plateau_len, remaining)
        plateau_noise = rng.normal(0, 3, plateau_end)
        boost[:plateau_end] = actual_height + plateau_noise
        
        # Phase 2: Gradual decay after plateau, but keep permanent boost
        if remaining > plateau_len:
            decay_len = remaining - plateau_len
            temp_boost = actual_height - permanent_boost
            decay_curve = temp_boost * np.exp(-decay_rate * np.arange(decay_len))
            boost[plateau_len:] = permanent_boost + decay_curve + rng.normal(0, 2, decay_len)
        
        y_updated[t_idx:] += boost
    
    return y_updated

# --- Define Async Weight Update Times (in data point indices) ---
# Only 3 updates, evenly spaced
update_indices = [450, 900, 1350]  # At ~7.5, 15, 22.5 minutes

# Convert indices to time in minutes for annotation
update_times_min = [time_minutes[idx] for idx in update_indices]

# --- Generate Throughput Data ---
y_base = generate_base_throughput(n_points, base_level=340, noise=5)
y_with_updates = add_async_weight_updates(y_base.copy(), update_indices, spike_height=55, plateau_len=200, decay_rate=0.006)

# --- Smooth with Moving Average (fix edge effects) ---
win = 25
kernel = np.ones(win) / win

# Pad data to avoid edge effects
pad_len = win * 2
y_base_padded = np.concatenate([y_base, np.full(pad_len, y_base[-50:].mean())])
y_updates_padded = np.concatenate([y_with_updates, np.full(pad_len, y_with_updates[-50:].mean())])

y_base_smooth = np.convolve(y_base_padded, kernel, mode="same")[:n_points]
y_updates_smooth = np.convolve(y_updates_padded, kernel, mode="same")[:n_points]

# Clip to reasonable range
y_base_smooth = np.clip(y_base_smooth, 280, 580)
y_updates_smooth = np.clip(y_updates_smooth, 280, 580)

# --- Create Beautiful Plot ---
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

# Set background
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')

# Color palette - modern and visually appealing
color_base = '#7f8c8d'      # Gray for baseline
color_async = '#e74c3c'     # Vibrant red for async updates
color_fill_base = '#bdc3c7'
color_fill_async = '#fadbd8'
color_accent = '#2ecc71'    # Green for update markers

# Plot baseline (without async updates) - dashed for comparison
ax.plot(time_minutes, y_base_smooth, 
        color=color_base, linewidth=2.5, linestyle='--', 
        alpha=0.8, label='Without Async Updates', zorder=2)

# Plot with async weight updates - solid, prominent
ax.plot(time_minutes, y_updates_smooth, 
        color=color_async, linewidth=3.0, 
        label='With Async Weight Updates', zorder=3)

# Add gradient fill under the async updates curve
ax.fill_between(time_minutes, 300, y_updates_smooth, 
                alpha=0.15, color=color_async, zorder=1)

# --- Mark Async Weight Update Events ---
for i, (t_min, t_idx) in enumerate(zip(update_times_min, update_indices)):
    y_val = y_updates_smooth[t_idx]
    
    # Vertical line at update point
    ax.axvline(x=t_min, color=color_accent, linestyle=':', 
               linewidth=1.2, alpha=0.6, zorder=1)
    
    # Diamond marker at the update point
    ax.scatter([t_min], [y_val], color=color_accent, s=80, 
               marker='D', zorder=5, edgecolors='white', linewidths=1.5)
    
    # Add label for all updates
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
ax.set_ylabel('Throughput (tokens/s)', fontweight='bold', labelpad=10)

# Title with shadow effect
title = ax.set_title('Throughput Over Time with Async Weight Updates', 
                     fontweight='bold', pad=20, fontsize=15)
title.set_path_effects([path_effects.withSimplePatchShadow(offset=(1, -1), alpha=0.1)])

# Grid styling
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Axis limits
ax.set_xlim(0, total_time_minutes + 0.5)
ax.set_ylim(290, 480)

# Legend with custom styling
legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                   framealpha=0.95, shadow=True, borderpad=0.6,
                   prop={'weight': 'bold', 'size': 9})
legend.get_frame().set_edgecolor('#888888')
legend.get_frame().set_linewidth(1.2)

# Add average throughput annotations
avg_base = np.mean(y_base_smooth)
avg_async = np.mean(y_updates_smooth)
improvement = ((avg_async - avg_base) / avg_base) * 100

stats_text = f'Baseline: {avg_base:.0f} t/s | Async: {avg_async:.0f} t/s | +{improvement:.1f}%'
ax.text(0.98, 0.03, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                 edgecolor='#c0392b', alpha=0.95, linewidth=1.5),
        color='#1a1a1a', fontweight='bold')

# Spine styling
for spine in ax.spines.values():
    spine.set_color('#cccccc')
    spine.set_linewidth(1)

# --- Save ---
out_path = "./throughput_async_weight_update.png"
plt.tight_layout()
plt.savefig(out_path, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"Figure saved to: {out_path}")
out_path
