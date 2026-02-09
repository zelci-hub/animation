#!/usr/bin/env python3
"""
Animated TTT Flow Diagram
Shows the dynamic flow of requests, drafting, verification, and async weight updates.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as path_effects

# --- Style Configuration ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'font.weight': 'bold',
})

# --- Colors ---
COLOR_BG = '#f8f9fa'
COLOR_USER = '#3498db'       # Blue - user/request
COLOR_DRAFT = '#e74c3c'      # Red - draft model
COLOR_TARGET = '#2ecc71'     # Green - target model
COLOR_TRAINER = '#9b59b6'    # Purple - trainer
COLOR_DATA = '#f39c12'       # Orange - data
COLOR_UPDATE = '#1abc9c'     # Teal - async update
COLOR_BOX = '#ecf0f1'
COLOR_ARROW = '#7f8c8d'

# --- Create Figure ---
fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.set_facecolor(COLOR_BG)
fig.patch.set_facecolor('white')
ax.axis('off')

# Title
title = ax.text(7, 7.6, 'Test-Time Training (TTT) Flow', fontsize=18, fontweight='bold',
                ha='center', va='center', color='#2c3e50')
title.set_path_effects([path_effects.withSimplePatchShadow(offset=(1, -1), alpha=0.1)])

# --- Static Elements ---

# Inference Server Box
inference_box = FancyBboxPatch((0.5, 3.5), 9, 3.5, boxstyle="round,pad=0.05,rounding_size=0.2",
                                facecolor='#e8f4f8', edgecolor='#3498db', linewidth=2, alpha=0.5)
ax.add_patch(inference_box)
ax.text(5, 6.7, 'Inference Server', fontsize=12, fontweight='bold', color='#2980b9', ha='center')

# Training Server Box
training_box = FancyBboxPatch((10, 3.5), 3.5, 3.5, boxstyle="round,pad=0.05,rounding_size=0.2",
                               facecolor='#f5eef8', edgecolor='#9b59b6', linewidth=2, alpha=0.5)
ax.add_patch(training_box)
ax.text(11.75, 6.7, 'Training Server', fontsize=12, fontweight='bold', color='#8e44ad', ha='center')

# User Input (left side)
user_box = FancyBboxPatch((0.3, 1.5), 1.5, 1.2, boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=COLOR_USER, edgecolor='#2980b9', linewidth=2, alpha=0.9)
ax.add_patch(user_box)
ax.text(1.05, 2.1, 'User\nInput', fontsize=9, fontweight='bold', color='white', ha='center', va='center')

# Response (right side of inference)
response_box = FancyBboxPatch((8, 1.5), 1.5, 1.2, boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='#27ae60', edgecolor='#1e8449', linewidth=2, alpha=0.9)
ax.add_patch(response_box)
ax.text(8.75, 2.1, 'Response', fontsize=9, fontweight='bold', color='white', ha='center', va='center')

# Draft Model boxes (3 times)
draft_positions = [(1.5, 4.5), (3.5, 4.5), (7.5, 4.5)]
draft_boxes = []
draft_labels = []
for i, (x, y) in enumerate(draft_positions):
    box = FancyBboxPatch((x, y), 1.3, 1.0, boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=COLOR_DRAFT, edgecolor='#c0392b', linewidth=2, alpha=0.3)
    ax.add_patch(box)
    draft_boxes.append(box)
    label = ax.text(x + 0.65, y + 0.5, f'Draft', fontsize=9, fontweight='bold', 
                    color='white', ha='center', va='center', alpha=0.3)
    draft_labels.append(label)

# Target Model box (in the middle)
target_box = FancyBboxPatch((5.5, 4.5), 1.3, 1.0, boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=COLOR_TARGET, edgecolor='#1e8449', linewidth=2, alpha=0.3)
ax.add_patch(target_box)
target_label = ax.text(6.15, 5.0, 'Target', fontsize=9, fontweight='bold', 
                        color='white', ha='center', va='center', alpha=0.3)

# Static arrows between draft/target
arrow_style = "Simple,tail_width=0.3,head_width=0.8,head_length=0.4"
static_arrows = [
    FancyArrowPatch((2.85, 5.0), (3.45, 5.0), arrowstyle=arrow_style, color=COLOR_ARROW, alpha=0.3),
    FancyArrowPatch((4.85, 5.0), (5.45, 5.0), arrowstyle=arrow_style, color=COLOR_ARROW, alpha=0.3),
    FancyArrowPatch((6.85, 5.0), (7.45, 5.0), arrowstyle=arrow_style, color=COLOR_ARROW, alpha=0.3),
]
for arrow in static_arrows:
    ax.add_patch(arrow)

# Data Buffer
data_buffer = FancyBboxPatch((10.5, 5.5), 1.0, 0.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=COLOR_DATA, edgecolor='#d68910', linewidth=1.5, alpha=0.3)
ax.add_patch(data_buffer)
data_label = ax.text(11, 5.85, 'Data\nBuffer', fontsize=7, fontweight='bold', 
                      color='white', ha='center', va='center', alpha=0.3)

# Trainer
trainer_box = FancyBboxPatch((10.5, 4.2), 1.0, 0.9, boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=COLOR_TRAINER, edgecolor='#7d3c98', linewidth=1.5, alpha=0.3)
ax.add_patch(trainer_box)
trainer_label = ax.text(11, 4.65, 'Trainer', fontsize=8, fontweight='bold', 
                         color='white', ha='center', va='center', alpha=0.3)

# New Speculator
speculator_box = FancyBboxPatch((12, 4.2), 1.2, 0.9, boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=COLOR_UPDATE, edgecolor='#16a085', linewidth=1.5, alpha=0.3)
ax.add_patch(speculator_box)
speculator_label = ax.text(12.6, 4.65, 'New\nSpec', fontsize=7, fontweight='bold', 
                            color='white', ha='center', va='center', alpha=0.3)

# Async Update arrow (curved, going back to draft models)
async_arrow = FancyArrowPatch((12.6, 4.1), (7.5, 3.8), 
                               connectionstyle="arc3,rad=-0.3",
                               arrowstyle=arrow_style, color=COLOR_UPDATE, 
                               linewidth=2, alpha=0.0)
ax.add_patch(async_arrow)
async_label = ax.text(10.5, 3.2, 'Async Update', fontsize=10, fontweight='bold',
                       color=COLOR_UPDATE, ha='center', alpha=0.0)

# --- Animation Elements ---

# Moving request dot
request_dot = Circle((0, 0), 0.15, facecolor=COLOR_USER, edgecolor='white', linewidth=2, zorder=10, alpha=0)
ax.add_patch(request_dot)

# Token dots (for drafting visualization)
token_dots = []
for i in range(6):
    dot = Circle((0, 0), 0.1, facecolor=COLOR_DRAFT, edgecolor='white', linewidth=1, zorder=10, alpha=0)
    ax.add_patch(dot)
    token_dots.append(dot)

# Status text
status_text = ax.text(7, 0.5, '', fontsize=11, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='#3498db', alpha=0.95, linewidth=1.5))

# --- Animation Logic ---
n_frames = 200
phase_frames = {
    'request_in': (0, 20),        # Request coming in
    'draft1': (20, 40),           # First draft
    'draft2': (40, 60),           # Second draft  
    'target_verify': (60, 85),    # Target verification
    'draft3': (85, 105),          # Third draft
    'response': (105, 125),       # Response out
    'data_transfer': (125, 145),  # Data to buffer
    'training': (145, 170),       # Training
    'async_update': (170, 200),   # Async update
}

def get_phase(frame):
    for phase, (start, end) in phase_frames.items():
        if start <= frame < end:
            return phase, (frame - start) / (end - start)
    return 'idle', 0

def animate(frame):
    phase, progress = get_phase(frame)
    
    # Reset all dynamic elements
    request_dot.set_alpha(0)
    for dot in token_dots:
        dot.set_alpha(0)
    
    # Reset box alphas
    for box, label in zip(draft_boxes, draft_labels):
        box.set_alpha(0.3)
        label.set_alpha(0.3)
    target_box.set_alpha(0.3)
    target_label.set_alpha(0.3)
    data_buffer.set_alpha(0.3)
    data_label.set_alpha(0.3)
    trainer_box.set_alpha(0.3)
    trainer_label.set_alpha(0.3)
    speculator_box.set_alpha(0.3)
    speculator_label.set_alpha(0.3)
    async_arrow.set_alpha(0)
    async_label.set_alpha(0)
    
    if phase == 'request_in':
        # Request dot moving from user to first draft
        x = 1.8 + progress * 0.4
        y = 2.1 + progress * 2.4
        request_dot.center = (x, y)
        request_dot.set_alpha(1)
        status_text.set_text('[1/8] User request incoming...')
        
    elif phase == 'draft1':
        # First draft activates
        draft_boxes[0].set_alpha(0.9)
        draft_labels[0].set_alpha(1)
        # Show tokens being generated
        n_tokens = int(progress * 3) + 1
        for i in range(min(n_tokens, 3)):
            token_dots[i].center = (2.2 + i * 0.3, 5.8)
            token_dots[i].set_alpha(0.8)
        status_text.set_text('[2/8] Draft #1: Generating tokens...')
        
    elif phase == 'draft2':
        # Second draft
        draft_boxes[0].set_alpha(0.9)
        draft_labels[0].set_alpha(1)
        draft_boxes[1].set_alpha(0.9)
        draft_labels[1].set_alpha(1)
        n_tokens = int(progress * 3) + 1
        for i in range(min(n_tokens, 3)):
            token_dots[i].center = (4.2 + i * 0.3, 5.8)
            token_dots[i].set_alpha(0.8)
        for arrow in static_arrows[:1]:
            arrow.set_alpha(0.9)
        status_text.set_text('[3/8] Draft #2: More tokens...')
        
    elif phase == 'target_verify':
        # Target verification
        draft_boxes[0].set_alpha(0.9)
        draft_boxes[1].set_alpha(0.9)
        draft_labels[0].set_alpha(1)
        draft_labels[1].set_alpha(1)
        target_box.set_alpha(0.9)
        target_label.set_alpha(1)
        for arrow in static_arrows[:2]:
            arrow.set_alpha(0.9)
        # Verification animation
        if progress < 0.5:
            status_text.set_text('[4/8] Target: Verifying tokens...')
        else:
            status_text.set_text('[4/8] Target: Tokens VERIFIED!')
            
    elif phase == 'draft3':
        # Third draft
        for box, label in zip(draft_boxes, draft_labels):
            box.set_alpha(0.9)
            label.set_alpha(1)
        target_box.set_alpha(0.9)
        target_label.set_alpha(1)
        for arrow in static_arrows:
            arrow.set_alpha(0.9)
        n_tokens = int(progress * 3) + 1
        for i in range(min(n_tokens, 3)):
            token_dots[i].center = (8.2 + i * 0.3, 5.8)
            token_dots[i].set_alpha(0.8)
        status_text.set_text('[5/8] Draft #3: Final tokens...')
        
    elif phase == 'response':
        # Response going out
        for box, label in zip(draft_boxes, draft_labels):
            box.set_alpha(0.9)
            label.set_alpha(1)
        target_box.set_alpha(0.9)
        target_label.set_alpha(1)
        for arrow in static_arrows:
            arrow.set_alpha(0.9)
        # Response dot
        x = 8.5 - progress * 0.3
        y = 4.5 - progress * 2.0
        request_dot.set_facecolor('#27ae60')
        request_dot.center = (x, y)
        request_dot.set_alpha(1)
        status_text.set_text('[6/8] Sending response to user!')
        
    elif phase == 'data_transfer':
        # Data flowing to training server
        for box, label in zip(draft_boxes, draft_labels):
            box.set_alpha(0.9)
            label.set_alpha(1)
        target_box.set_alpha(0.9)
        target_label.set_alpha(1)
        data_buffer.set_alpha(0.9)
        data_label.set_alpha(1)
        # Data dots moving
        for i in range(3):
            phase_offset = (progress + i * 0.15) % 1
            x = 9 + phase_offset * 1.5
            y = 5.0 + phase_offset * 0.8
            token_dots[i].set_facecolor(COLOR_DATA)
            token_dots[i].center = (x, y)
            token_dots[i].set_alpha(0.8 * (1 - phase_offset))
        status_text.set_text('[7/8] Collecting data for training...')
        
    elif phase == 'training':
        # Training happening
        for box, label in zip(draft_boxes, draft_labels):
            box.set_alpha(0.9)
            label.set_alpha(1)
        target_box.set_alpha(0.9)
        target_label.set_alpha(1)
        data_buffer.set_alpha(0.9)
        data_label.set_alpha(1)
        trainer_box.set_alpha(0.9)
        trainer_label.set_alpha(1)
        # Pulsing effect on trainer
        pulse = 0.7 + 0.3 * np.sin(progress * 4 * np.pi)
        trainer_box.set_alpha(pulse)
        if progress > 0.5:
            speculator_box.set_alpha(0.9)
            speculator_label.set_alpha(1)
        status_text.set_text('[7/8] Training new speculator...')
        
    elif phase == 'async_update':
        # Async update back to draft models
        for box, label in zip(draft_boxes, draft_labels):
            box.set_alpha(0.9)
            label.set_alpha(1)
        target_box.set_alpha(0.9)
        target_label.set_alpha(1)
        data_buffer.set_alpha(0.9)
        data_label.set_alpha(1)
        trainer_box.set_alpha(0.9)
        trainer_label.set_alpha(1)
        speculator_box.set_alpha(0.9)
        speculator_label.set_alpha(1)
        
        # Async arrow appears
        async_arrow.set_alpha(min(1, progress * 2))
        async_label.set_alpha(min(1, progress * 2))
        
        # Flash effect on draft boxes
        if progress > 0.5:
            flash = 0.9 + 0.1 * np.sin((progress - 0.5) * 8 * np.pi)
            for box in draft_boxes:
                box.set_facecolor(COLOR_UPDATE if int(progress * 10) % 2 == 0 else COLOR_DRAFT)
            status_text.set_text('[8/8] ASYNC UPDATE: Draft models upgraded!')
        else:
            status_text.set_text('[8/8] Sending weights to inference...')
    
    else:
        status_text.set_text('Ready for next request...')
    
    return []

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=80, blit=False)

# Save as GIF
out_path = "./ttt_flow_animated.gif"
print(f"Saving animation to: {out_path}")
print("This may take a minute...")

anim.save(out_path, writer='pillow', fps=15)
plt.close()

print(f"Animation saved to: {out_path}")
