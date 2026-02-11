#!/usr/bin/env python3
"""
Aurora TTT Flow Diagram - Animated
Shows three cycles:
1. Nurse uses medical speculator (happy - works well)
2. Doctor uses academic speculator (sad - not trained yet)
3. Doctor returns after online training (happy - Aurora trained it)
"""

import os
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, Polygon, FancyArrowPatch, Rectangle
import matplotlib.patheffects as path_effects
from matplotlib import image as mimage

# --- Style Configuration ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'font.weight': 'bold',
})

# --- Colors ---
COLOR_BG = '#f0f4f8'
COLOR_NURSE = '#ec4899'       # Pink for nurse
COLOR_DOCTOR = '#3b82f6'      # Blue for doctor
COLOR_DRAFT = '#f97316'       # Orange - draft/speculator
COLOR_TARGET = '#10b981'      # Green - target model
COLOR_DATA = '#8b5cf6'        # Purple - data buffer
# Light/pale versions for "replay" balls (e.g. in Status 4)
COLOR_DRAFT_LIGHT = '#fed7aa'
COLOR_TARGET_LIGHT = '#a7f3d0'
COLOR_HAPPY = '#22c55e'       # Green - happy
COLOR_SAD = '#ef4444'         # Red - sad
COLOR_AURORA = '#6366f1'      # Indigo - Aurora theme
COLOR_SERVER = '#e2e8f0'

# --- Optional: custom user images (JPG/PNG)，按 cycle 使用 ---
# cycle 1 → nurse, cycle 2 → student1, cycle 3 → student2
NURSE_IMAGE_PATH = '/data/zshao/animation/nurse.png'      # cycle 1
STUDENT1_IMAGE_PATH = '/data/zshao/animation/student1.png'   # cycle 2
STUDENT2_IMAGE_PATH = '/data/zshao/animation/student2.png'   # cycle 3
DOCTOR_IMAGE_PATH = None   # fallback 当无 cycle 专用图时
_user_image_cache = {}   # path -> ndarray

def _load_image(path):
    """Load image once and cache by path. Returns None if path is None or file missing."""
    global _user_image_cache
    if path is None or not (path and str(path).strip()):
        return None
    path = os.path.abspath(os.path.expanduser(str(path).strip()))
    if not os.path.isfile(path):
        return None
    if path in _user_image_cache:
        return _user_image_cache[path]
    try:
        img = mimage.imread(path)
        _user_image_cache[path] = img
        return img
    except Exception:
        return None

def _load_nurse_image(path):
    return _load_image(path)

def _load_doctor_image(path):
    return _load_image(path)

# --- Create Figure ---
fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_facecolor(COLOR_BG)
fig.patch.set_facecolor('white')
ax.axis('off')

# Aurora Title with gradient effect（标题/副标题稍向右）
AURORA_LOGO_PATH = '/data/zshao/animation/finalaurora.png'
TITLE_X = 7.0          # 再右移
TITLE_Y = 8.9
SUBTITLE_Y = 8.3
title = ax.text(TITLE_X, TITLE_Y, 'Aurora', fontsize=32, fontweight='bold',
                ha='center', va='center', color=COLOR_AURORA,
                style='italic')
title.set_path_effects([
    path_effects.withStroke(linewidth=4, foreground='white'),
    path_effects.withSimplePatchShadow(offset=(2, -2), alpha=0.15)
])
subtitle = ax.text(TITLE_X, SUBTITLE_Y, 'Test-Time Training for Speculative Decoding', 
                   fontsize=14, ha='center', va='center', color='#64748b')

# --- Static Elements ---
# Inference Server：垂直方向加高，以容纳 Draft/Target 及更大间距
INFERENCE_SERVER_RIGHT = 9.2
INFERENCE_SERVER_LEFT = 4.5   # 与 Input/Output 右缘 3.25 留隙
INFERENCE_SERVER_W = INFERENCE_SERVER_RIGHT - INFERENCE_SERVER_LEFT
INFERENCE_SERVER_Y = 2.5
INFERENCE_SERVER_H = 5.0   # 加高
server_box = FancyBboxPatch((INFERENCE_SERVER_LEFT, INFERENCE_SERVER_Y), INFERENCE_SERVER_W, INFERENCE_SERVER_H,
                             boxstyle="round,pad=0.08,rounding_size=0.25",
                             facecolor=COLOR_SERVER, edgecolor='#94a3b8', linewidth=2, alpha=0.7)
ax.add_patch(server_box)
# Server 标签随框加高上移：框顶 = INFERENCE_SERVER_Y + INFERENCE_SERVER_H，标签在框顶略下
INFERENCE_SERVER_LABEL_Y = INFERENCE_SERVER_Y + INFERENCE_SERVER_H - 0.3
ax.text(INFERENCE_SERVER_LEFT + INFERENCE_SERVER_W / 2, INFERENCE_SERVER_LABEL_Y, 'Inference Server', fontsize=16, fontweight='bold', color='#475569', ha='center')

# Training Server Box：垂直方向与 Inference Server 同高
TRAIN_SERVER_X, TRAIN_SERVER_W = 11, 3.8
TRAIN_SERVER_Y = 2.5
TRAIN_SERVER_H = 5.0
train_box = FancyBboxPatch((TRAIN_SERVER_X, TRAIN_SERVER_Y), TRAIN_SERVER_W, TRAIN_SERVER_H, boxstyle="round,pad=0.08,rounding_size=0.25",
                            facecolor='#faf5ff', edgecolor=COLOR_DATA, linewidth=2, alpha=0.7)
ax.add_patch(train_box)
TRAIN_SERVER_LABEL_Y = TRAIN_SERVER_Y + TRAIN_SERVER_H - 0.3
ax.text(12.9, TRAIN_SERVER_LABEL_Y, 'Training Server', fontsize=16, fontweight='bold', color='#7c3aed', ha='center')
# Logo 放在 Training Server 上方，大小为原来的 2 倍
_logo_img = _load_image(AURORA_LOGO_PATH)
if _logo_img is not None:
    _logo_h = 6/5   # 缩小为原来的 2/3
    _logo_w = _logo_h * (_logo_img.shape[1] / _logo_img.shape[0]) if _logo_img.shape[0] else _logo_h
    _logo_center_x = 11.9   # 左移一点（原与 Training Server 中心 12.9 对齐）
    _train_top = TRAIN_SERVER_Y + TRAIN_SERVER_H   # 7.5
    _logo_center_y = 8.4   # 再下移一点
    ax.imshow(_logo_img, extent=[_logo_center_x - _logo_w/2, _logo_center_x + _logo_w/2,
                                 _logo_center_y - _logo_h/2, _logo_center_y + _logo_h/2],
              aspect='equal', zorder=5, interpolation='bilinear')

# Data Buffer：参考图 — 高度约为 Draft/Target 的 70%，上缘明显低于 Target/Trainer 底(4.0)
DATA_BUFFER_LEFT = 8.5
DATA_BUFFER_W = 3.2
DATA_BUFFER_BOTTOM = 2.8   # 上移：上缘 3.5，与 Target 底 4.0 留隙
DATA_BUFFER_H = 0.7
DATA_BUFFER_CX = DATA_BUFFER_LEFT + DATA_BUFFER_W / 2
DATA_BUFFER_CY = DATA_BUFFER_BOTTOM + DATA_BUFFER_H / 2
data_buffer = FancyBboxPatch((DATA_BUFFER_LEFT, DATA_BUFFER_BOTTOM), DATA_BUFFER_W, DATA_BUFFER_H,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor=COLOR_DATA, edgecolor='#7c3aed', linewidth=2, alpha=0.8)
ax.add_patch(data_buffer)
# Data Buffer 进度条填充：内嵌矩形，高度表示“满度”，在 animate 中按 status 更新
DATA_BUFFER_MARGIN = 0.08
_data_buffer_inner_w = DATA_BUFFER_W - 2 * DATA_BUFFER_MARGIN
_data_buffer_inner_h_max = DATA_BUFFER_H - 2 * DATA_BUFFER_MARGIN
data_buffer_fill = Rectangle((DATA_BUFFER_LEFT + DATA_BUFFER_MARGIN, DATA_BUFFER_BOTTOM + DATA_BUFFER_MARGIN),
                              _data_buffer_inner_w, _data_buffer_inner_h_max * 0.5,
                              facecolor='white', edgecolor='none', alpha=0.4, zorder=1)
ax.add_patch(data_buffer_fill)
ax.text(DATA_BUFFER_CX, DATA_BUFFER_CY, 'Data\nBuffer', fontsize=10, fontweight='bold', color='white', ha='center', va='center', zorder=2)

# 统一 Draft/Target/Trainer/New Spec：高度为原来 2/3，宽度为原来 2 倍
MODEL_BOX_H = 1.0 * (2 / 3)   # 2/3
# Inference Server：Draft 上、Target 下（同 x 中心竖排），间距加大
DRAFT_W = 1.4 * 2   # 2.8
INFERENCE_CENTER_X = 6.85   # Draft 与 Target 共用中心 x
DRAFT_X = INFERENCE_CENTER_X - DRAFT_W / 2
DRAFT_Y = 5.5   # Draft 在上（在加高后的 server 内偏上）
GAP_DRAFT_TARGET = 1.0   # Draft 与 Target 之间间距
TARGET_Y = DRAFT_Y - MODEL_BOX_H - GAP_DRAFT_TARGET   # Target 在下
DRAFT_CENTER_Y = DRAFT_Y + MODEL_BOX_H / 2
TARGET_CENTER_Y = TARGET_Y + MODEL_BOX_H / 2
draft_positions = [(DRAFT_X, DRAFT_Y)]
draft_boxes = []
draft_labels = []
draft_med_icons = []
draft_res_icons = []
for i, (x, y) in enumerate(draft_positions):
    box = FancyBboxPatch((x, y), DRAFT_W, MODEL_BOX_H, boxstyle="round,pad=0.05,rounding_size=0.15",
                          facecolor=COLOR_DRAFT, edgecolor='#ea580c', linewidth=2, alpha=0.3)
    ax.add_patch(box)
    draft_boxes.append(box)
    label = ax.text(x + DRAFT_W / 2, y + MODEL_BOX_H / 2, f'Draft', fontsize=11, fontweight='bold', 
                    color='white', ha='center', va='center', alpha=0.3)
    draft_labels.append(label)
    
    # Health icon: "+Health" in white circle（在框上方）
    med_x = x + DRAFT_W * 0.32
    res_x = x + DRAFT_W * 0.68
    med_circle = Circle((med_x, y + MODEL_BOX_H + 0.2), 0.28, facecolor='white',
                         edgecolor='#ef4444', linewidth=2, alpha=0, zorder=15)
    ax.add_patch(med_circle)
    med_cross = ax.text(med_x, y + MODEL_BOX_H + 0.2, '+Health', fontsize=8, fontweight='bold',
                        color='#ef4444', ha='center', va='center', alpha=0, zorder=16)
    draft_med_icons.append((med_circle, med_cross))
    
    # Math icon: "+Math" in blue circle
    res_circle = Circle((res_x, y + MODEL_BOX_H + 0.2), 0.28, facecolor='white',
                         edgecolor='#3b82f6', linewidth=2, alpha=0, zorder=15)
    ax.add_patch(res_circle)
    res_star = ax.text(res_x, y + MODEL_BOX_H + 0.2, '+Math', fontsize=8, fontweight='bold',
                       color='#3b82f6', ha='center', va='center', alpha=0, zorder=16)
    draft_res_icons.append((res_circle, res_star))

# Target Model（在 Draft 下方）
target_box = FancyBboxPatch((DRAFT_X, TARGET_Y), DRAFT_W, MODEL_BOX_H, boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=COLOR_TARGET, edgecolor='#059669', linewidth=2, alpha=0.3)
ax.add_patch(target_box)
target_label = ax.text(INFERENCE_CENTER_X, TARGET_CENTER_Y, 'Target', fontsize=11, fontweight='bold', 
                        color='white', ha='center', va='center', alpha=0.3)

# Training Server：New Spec 上、Trainer 下（同 x 中心竖排），间距加大
TRAIN_CENTER_X = 12.9
SPEC_BOX_W = 1.4 * 2   # 2.8
SPEC_BOX_H = MODEL_BOX_H
SPEC_X = TRAIN_CENTER_X - SPEC_BOX_W / 2
SPEC_Y = 5.5   # New Spec 在上（与 Draft 同高）
GAP_SPEC_TRAINER = 1.0   # New Spec 与 Trainer 之间间距
TRAINER_Y = SPEC_Y - MODEL_BOX_H - GAP_SPEC_TRAINER   # Trainer 在下
TRAINER_W = 1.3 * 2   # 2.6
TRAINER_X = TRAIN_CENTER_X - TRAINER_W / 2
SPEC_CENTER_Y = SPEC_Y + SPEC_BOX_H / 2
TRAINER_CENTER_Y = TRAINER_Y + MODEL_BOX_H / 2
trainer_box = FancyBboxPatch((TRAINER_X, TRAINER_Y), TRAINER_W, MODEL_BOX_H, boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor=COLOR_AURORA, edgecolor='#4f46e5', linewidth=2, alpha=0.3)
ax.add_patch(trainer_box)
trainer_label = ax.text(TRAIN_CENTER_X, TRAINER_CENTER_Y, 'Trainer', fontsize=11, fontweight='bold',
                         color='white', ha='center', va='center', alpha=0.3)
spec_box = FancyBboxPatch((SPEC_X, SPEC_Y), SPEC_BOX_W, SPEC_BOX_H, boxstyle="round,pad=0.05,rounding_size=0.15",
                           facecolor=COLOR_DRAFT, edgecolor='#ea580c', linewidth=2, alpha=0.3)
ax.add_patch(spec_box)
spec_label = ax.text(TRAIN_CENTER_X, SPEC_CENTER_Y, 'New\nSpec', fontsize=10, fontweight='bold',
                      color='white', ha='center', va='center', alpha=0.3)

# --- Draw cute user function ---
USER_IMAGE_SIZE = 0.9   # 头像再缩小
USER_IMAGE_SLIDE = 2.0  # 切入时从左向右滑动的距离

def draw_user(ax, x, y, user_type, is_happy, alpha=1.0, cycle=None, entrance_progress=1.0):
    """Draw a cute, polished user icon (nurse/doctor/student). entrance_progress 0→1 时从左到右切入."""
    elements = []
    skin = '#fde68a'           # Warmer skin tone
    skin_edge = '#f59e0b'
    shadow_color = 'black'

    # Body color & accent
    if user_type == 'nurse':
        body_color = COLOR_NURSE
        hat_color = '#fce7f3'
        collar_color = '#fdf2f8'
    else:
        body_color = COLOR_DOCTOR
        hat_color = '#dbeafe'
        collar_color = '#eff6ff'

    def _draw_user_image(img, x, y, alpha, entrance_progress):
        h = USER_IMAGE_SIZE
        cy = y + 0.25
        # 从左到右切入: entrance_progress=0 时在左侧，=1 时在最终位置
        t = min(1.0, entrance_progress)
        left = x - h/2 - (1 - t) * USER_IMAGE_SLIDE
        right = left + h
        bottom, top = cy - h/2, cy + h/2
        im_artist = ax.imshow(img, extent=[left, right, bottom, top], aspect='equal',
                              alpha=alpha, zorder=10, interpolation='bilinear')
        return im_artist

    # --- 按 cycle 使用对应 PNG：cycle1→nurse, cycle2→student1, cycle3→student2 ---
    if cycle == 1 and NURSE_IMAGE_PATH:
        img = _load_image(NURSE_IMAGE_PATH)
        if img is not None:
            elements.append(_draw_user_image(img, x, y, alpha, entrance_progress))
            return elements
    if cycle == 2 and STUDENT1_IMAGE_PATH:
        img = _load_image(STUDENT1_IMAGE_PATH)
        if img is not None:
            elements.append(_draw_user_image(img, x, y, alpha, entrance_progress))
            return elements
    if cycle == 3 and STUDENT2_IMAGE_PATH:
        img = _load_image(STUDENT2_IMAGE_PATH)
        if img is not None:
            elements.append(_draw_user_image(img, x, y, alpha, entrance_progress))
            return elements

    # --- Fallback: 按 user_type 用 doctor/nurse 图（若上面未命中）---
    if str(user_type).lower() == 'nurse' and NURSE_IMAGE_PATH:
        img = _load_image(NURSE_IMAGE_PATH)
        if img is not None:
            elements.append(_draw_user_image(img, x, y, alpha, entrance_progress))
            return elements
    if str(user_type).lower() == 'doctor' and DOCTOR_IMAGE_PATH:
        img = _load_doctor_image(DOCTOR_IMAGE_PATH)
        if img is not None:
            elements.append(_draw_user_image(img, x, y, alpha, entrance_progress))
            return elements
        # else: file missing or failed to load → fall through to cartoon doctor

    # 1) Soft shadow under head (depth)
    shadow = Circle((x + 0.08, y + 0.08), 0.32, facecolor=shadow_color,
                    edgecolor='none', alpha=0.12 * alpha, zorder=9)
    ax.add_patch(shadow)
    elements.append(shadow)

    # 2) Body (rounded, slightly wider at bottom)
    body = FancyBboxPatch((x - 0.32, y - 0.52), 0.64, 0.54,
                          boxstyle="round,pad=0.04,rounding_size=0.12",
                          facecolor=body_color, edgecolor='white', linewidth=2.2,
                          alpha=alpha, zorder=10)
    ax.add_patch(body)
    elements.append(body)

    # 3) Collar (white/light strip at top of body)
    collar = FancyBboxPatch((x - 0.22, y - 0.08), 0.44, 0.12,
                            boxstyle="round,pad=0.02,rounding_size=0.06",
                            facecolor=collar_color, edgecolor='white', linewidth=1.5,
                            alpha=alpha, zorder=11)
    ax.add_patch(collar)
    elements.append(collar)

    # 4) Neck (connects head and body)
    neck = FancyBboxPatch((x - 0.12, y - 0.02), 0.24, 0.1,
                          boxstyle="round,pad=0.01", facecolor=skin, edgecolor=skin_edge,
                          linewidth=1.2, alpha=alpha, zorder=10)
    ax.add_patch(neck)
    elements.append(neck)

    # 5) Head with subtle highlight (glossy)
    head = Circle((x, y + 0.32), 0.34, facecolor=skin, edgecolor=skin_edge,
                  linewidth=2.2, alpha=alpha, zorder=10)
    ax.add_patch(head)
    elements.append(head)
    highlight = Circle((x - 0.1, y + 0.45), 0.1, facecolor='white', edgecolor='none',
                       alpha=0.35 * alpha, zorder=11)
    ax.add_patch(highlight)
    elements.append(highlight)

    # 6) Hat (nurse) or glasses (doctor)
    if user_type == 'nurse':
        # Nurse cap (folded wing shape - two overlapping rectangles)
        hat = FancyBboxPatch((x - 0.28, y + 0.52), 0.56, 0.22,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor=hat_color, edgecolor=COLOR_NURSE, linewidth=1.8,
                             alpha=alpha, zorder=11)
        ax.add_patch(hat)
        elements.append(hat)
        cross_h, = ax.plot([x - 0.12, x + 0.12], [y + 0.64, y + 0.64],
                           color=COLOR_NURSE, linewidth=2.5, alpha=alpha, zorder=12)
        cross_v, = ax.plot([x, x], [y + 0.54, y + 0.74],
                          color=COLOR_NURSE, linewidth=2.5, alpha=alpha, zorder=12)
        elements.extend([cross_h, cross_v])
    else:
        # Doctor glasses with bridge
        glass_l = Circle((x - 0.13, y + 0.35), 0.095, facecolor='none',
                         edgecolor='#1e3a5f', linewidth=1.8, alpha=alpha, zorder=11)
        glass_r = Circle((x + 0.13, y + 0.35), 0.095, facecolor='none',
                         edgecolor='#1e3a5f', linewidth=1.8, alpha=alpha, zorder=11)
        ax.add_patch(glass_l)
        ax.add_patch(glass_r)
        elements.extend([glass_l, glass_r])
        bridge, = ax.plot([x - 0.04, x + 0.04], [y + 0.36, y + 0.36],
                         color='#1e3a5f', linewidth=1.5, alpha=alpha, zorder=11)
        elements.append(bridge)
        # Small reflection on glasses
        refl_l = Circle((x - 0.16, y + 0.38), 0.03, facecolor='white', edgecolor='none', alpha=0.5 * alpha, zorder=12)
        refl_r = Circle((x + 0.10, y + 0.38), 0.03, facecolor='white', edgecolor='none', alpha=0.5 * alpha, zorder=12)
        ax.add_patch(refl_l)
        ax.add_patch(refl_r)
        elements.extend([refl_l, refl_r])

    # 7) Rosy cheeks
    cheek_l = Circle((x - 0.22, y + 0.28), 0.06, facecolor='#fda4af', edgecolor='none', alpha=0.5 * alpha, zorder=11)
    cheek_r = Circle((x + 0.22, y + 0.28), 0.06, facecolor='#fda4af', edgecolor='none', alpha=0.5 * alpha, zorder=11)
    ax.add_patch(cheek_l)
    ax.add_patch(cheek_r)
    elements.extend([cheek_l, cheek_r])

    # 8) Eyebrows + eyes + mouth
    if is_happy:
        # Curved up eyebrows
        theta = np.linspace(0, np.pi, 15)
        brow_y_offset = 0.42 + 0.04 * np.sin(theta)
        brow_l_x = x - 0.1 - 0.06 * np.cos(theta)
        brow_r_x = x + 0.1 + 0.06 * np.cos(theta)
        line_l, = ax.plot(brow_l_x, brow_y_offset, color='#78350f', linewidth=1.8, alpha=alpha, zorder=11)
        line_r, = ax.plot(brow_r_x, brow_y_offset, color='#78350f', linewidth=1.8, alpha=alpha, zorder=11)
        elements.extend([line_l, line_r])
        # Eyes with sparkle
        eye_l = ax.scatter([x - 0.1], [y + 0.38], s=28, c='#1f2937', alpha=alpha, zorder=11, edgecolors='none')
        eye_r = ax.scatter([x + 0.1], [y + 0.38], s=28, c='#1f2937', alpha=alpha, zorder=11, edgecolors='none')
        spark_l = ax.scatter([x - 0.12], [y + 0.40], s=8, c='white', alpha=alpha, zorder=12)
        spark_r = ax.scatter([x + 0.08], [y + 0.40], s=8, c='white', alpha=alpha, zorder=12)
        elements.extend([eye_l, eye_r, spark_l, spark_r])
        # Smile (smooth arc)
        smile = Wedge((x, y + 0.22), 0.14, 205, 335, width=0.055, facecolor=COLOR_HAPPY, alpha=alpha, zorder=11)
        ax.add_patch(smile)
        elements.append(smile)
    else:
        # Curved down eyebrows (sad)
        theta = np.linspace(0, np.pi, 15)
        brow_y_offset = 0.46 - 0.03 * np.sin(theta)
        brow_l_x = x - 0.1 - 0.06 * np.cos(theta)
        brow_r_x = x + 0.1 + 0.06 * np.cos(theta)
        line_l, = ax.plot(brow_l_x, brow_y_offset, color='#78350f', linewidth=1.8, alpha=alpha, zorder=11)
        line_r, = ax.plot(brow_r_x, brow_y_offset, color='#78350f', linewidth=1.8, alpha=alpha, zorder=11)
        elements.extend([line_l, line_r])
        eye_l = ax.scatter([x - 0.1], [y + 0.38], s=24, c='#1f2937', alpha=alpha, zorder=11, marker='v')
        eye_r = ax.scatter([x + 0.1], [y + 0.38], s=24, c='#1f2937', alpha=alpha, zorder=11, marker='v')
        elements.extend([eye_l, eye_r])
        frown = Wedge((x, y + 0.12), 0.12, 25, 155, width=0.045, facecolor=COLOR_SAD, alpha=alpha, zorder=11)
        ax.add_patch(frown)
        elements.append(frown)

    return elements

# --- Response bubble ---
def draw_response_bubble(ax, x, y, is_good, alpha=1.0):
    """Draw a cute response bubble"""
    # Bubble
    bubble = FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8,
                             boxstyle="round,pad=0.1,rounding_size=0.2",
                             facecolor='white', edgecolor='#94a3b8', 
                             linewidth=2, alpha=alpha, zorder=10)
    ax.add_patch(bubble)
    
    # Content
    if is_good:
        text = ax.text(x, y, 'Fast!', fontsize=12, fontweight='bold',
                       color=COLOR_HAPPY, ha='center', va='center', alpha=alpha, zorder=11)
    else:
        text = ax.text(x, y, 'Slow...', fontsize=12, fontweight='bold',
                       color=COLOR_SAD, ha='center', va='center', alpha=alpha, zorder=11)
    
    return [bubble, text]

# --- Animation State ---
SPEC_FLY_START = (SPEC_X, SPEC_Y)
SPEC_FLY_END = (DRAFT_X, DRAFT_Y)

class AnimState:
    def __init__(self):
        self.user_elements = []
        self.response_elements = []
        self.data_dots = []
        self.spec_fly_elements = []  # flying New Spec copy (box + label)

state = AnimState()

def draw_dots_along_path(ax, x0, y0, x1, y1, progress, n_dots=5, color=COLOR_DATA, dot_radius=0.1, state_list=None):
    """Draw n_dots small circles moving from (x0,y0) toward (x1,y1). progress 0~1: wave advances from start to end."""
    if state_list is None:
        state_list = state.data_dots
    for i in range(n_dots):
        # t in [0,1): dot position along path. Use (1 - t) so that as progress increases,
        # the wave visibly moves from (x0,y0) to (x1,y1) (not the other way around).
        t_raw = (progress * (n_dots + 2) + i * 0.2) % 1.0
        t = 1.0 - t_raw
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        dot = Circle((x, y), dot_radius, facecolor=color, edgecolor='none', alpha=0.85, zorder=14)
        ax.add_patch(dot)
        state_list.append(dot)


def draw_dots_batch_along_path(ax, x0, y0, x1, y1, batch_progress, n_dots=3, color=COLOR_DATA, dot_radius=0.1, state_list=None, alpha=0.85):
    """每次发 n_dots 个小球，沿路径从 (x0,y0) 到 (x1,y1)；batch_progress 0~1 表示本批小球从起点到终点的进度。
    只画本批的 n_dots 个球，略错开（0.2 间距），到达终点后再发下一批由外部用 progress 分段实现。"""
    if state_list is None:
        state_list = state.data_dots
    for i in range(n_dots):
        # 本批内第 i 个球：略滞后，t = batch_progress - i * 0.2，且只在 [0,1] 内绘制（未出发或已到达的不画）
        t = batch_progress - i * 0.2
        if t < 0 or t > 1:
            continue
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        dot = Circle((x, y), dot_radius, facecolor=color, edgecolor='none', alpha=alpha, zorder=14)
        ax.add_patch(dot)
        state_list.append(dot)

# Status text 与 cycle text 的基准字号（出场动画在 TEXT_HOLD_FRAMES 内先放大再缩回）
STATUS_TEXT_BASE_FONTSIZE = 14
CYCLE_TEXT_BASE_FONTSIZE = 12
status_text = ax.text(8, 1.2, '', fontsize=STATUS_TEXT_BASE_FONTSIZE, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor=COLOR_AURORA, alpha=0.95, linewidth=2))

# Cycle indicator
cycle_text = ax.text(8, 0.5, '', fontsize=CYCLE_TEXT_BASE_FONTSIZE, ha='center', va='center', color='#64748b')

# --- User position：头像缩小并上移
USER_X = 1.5
USER_Y = 6.9   # 头像再上移，在 Input 框上方

# --- Input/Output dialog content per cycle (nurse, student1, student2) ---
INPUT_OUTPUT_BY_CYCLE = [
    # Cycle 1 - Nurse
    (
        "Nurse: Hi , what are the common side effects of amoxicillin?",
        "Assistant: Common side effects include nausea, diarrhea, and mild rash, but severe "
        "symptoms like trouble breathing may indicate an allergic reaction."
    ),
    # Cycle 2 - Student (chickens & rabbits)
    (
        "Student: A cage has 20 animals, some chickens, some rabbits. Total legs: 56. How many of each? ",
        "Assistant: Let chickens be (c) and rabbits be (r). Then (c + r = 20), and (2c + 4r = 56). "
        "Subtract (2(c + r) = 40) from 56 to get (2r = 16), so (r = 8), and (c = 12)."
    ),
    # Cycle 3 - Student (runners)
    (
        "Student: A runner starts at 6 km/h. Ten minutes later, another follows at 9 km/h. When will they meet?",
        "Assistant: The first runner gets a head start of (6 x 10/60 = 1) km. The speed difference "
        "is (9 - 6 = 3) km/h, so catch-up time is (1 ÷ 3 = 1/3) hour = 20 minutes."
    ),
]
WRAP_WIDTH = 36
INPUT_WRAP_WIDTH = 30  # Input 框内每行字符数更少，避免水平方向超出框

# Input/Output: 整体下移；Output 约为 Input 高度 2 倍，Output 顶与 Draft 中心(4.5)对齐
INPUT_BOX_H = 1.0
OUTPUT_BOX_H = 2.0
OUTPUT_BOX_Y = 3.2   # Output / Input 整体上移
INPUT_BOX_Y = OUTPUT_BOX_Y + OUTPUT_BOX_H + 0.25   # Input 在 Output 上方
# Input/Output 框：左边和上下不变，右边右移 → 宽 3.0
INPUT_BOX_LEFT, INPUT_BOX_W = 0.25, 3.0
OUTPUT_BOX_LEFT, OUTPUT_BOX_W = 0.25, 3.0
INPUT_PAD_TOP, INPUT_PAD_LEFT = 0.22, 0.12
OUTPUT_PAD_TOP, OUTPUT_PAD_LEFT = 0.22, 0.12
input_box = FancyBboxPatch((INPUT_BOX_LEFT, INPUT_BOX_Y), INPUT_BOX_W, INPUT_BOX_H, boxstyle="round,pad=0.06,rounding_size=0.15",
                            facecolor='#f1f5f9', edgecolor='#94a3b8', linewidth=1.5, alpha=0.95, zorder=8)
ax.add_patch(input_box)
input_content = ax.text(INPUT_BOX_LEFT + INPUT_PAD_LEFT, INPUT_BOX_Y + INPUT_BOX_H - INPUT_PAD_TOP, '', fontsize=8, ha='left', va='top',
                         color='#334155', zorder=9)
output_box = FancyBboxPatch((OUTPUT_BOX_LEFT, OUTPUT_BOX_Y), OUTPUT_BOX_W, OUTPUT_BOX_H, boxstyle="round,pad=0.06,rounding_size=0.15",
                            facecolor='#f0fdf4', edgecolor='#86efac', linewidth=1.5, alpha=0.95, zorder=8)
ax.add_patch(output_box)
output_content = ax.text(OUTPUT_BOX_LEFT + OUTPUT_PAD_LEFT, OUTPUT_BOX_Y + OUTPUT_BOX_H - OUTPUT_PAD_TOP, '', fontsize=8, ha='left', va='top',
                         color='#14532d', zorder=9)

# --- Animation Parameters ---
# 每进入新 cycle/status：先换字 → TEXT_HOLD_FRAMES 做“先放大再缩回” → PAUSE_AFTER_ENTRANCE_FRAMES 暂停 → 再执行动画
TEXT_HOLD_FRAMES = 10
PAUSE_AFTER_ENTRANCE_FRAMES = 10
PHASE_ENTRANCE_FRAMES = TEXT_HOLD_FRAMES + PAUSE_AFTER_ENTRANCE_FRAMES   # 20：缩放 + 暂停
# 各 phase 内容帧数（不含 entrance）；小球速度为原来 1/2，故与小球相关的帧数加倍
C1_INPUT_TYPING, C1_INPUT_TO_DRAFT = 25, 20   # INPUT_TO_DRAFT 10→20
C1_GENERATION, C1_BUFFER, C1_SPEC_FLY = 65, 40, 20   # C1_BUFFER 20→40, C1_SPEC_FLY 10→20
FRAMES_GENERATION_C2 = 125
C2_INPUT_TYPING, C2_INPUT_TO_DRAFT = 25, 20   # INPUT_TO_DRAFT 10→20
C2_BUFFER, C2_SPEC_FLY = 40, 20   # C2_BUFFER 20→40, C2_SPEC_FLY 10→20
# 每 phase 总帧数 = PHASE_ENTRANCE_FRAMES + 内容帧数
# Cycle 1 & 3 各 phase 累计起始帧
C1_P1 = PHASE_ENTRANCE_FRAMES + C1_INPUT_TYPING
C1_P2 = C1_P1 + PHASE_ENTRANCE_FRAMES + C1_INPUT_TO_DRAFT
C1_P3 = C1_P2 + PHASE_ENTRANCE_FRAMES + C1_GENERATION
C1_P4 = C1_P3 + PHASE_ENTRANCE_FRAMES + C1_BUFFER
C1_P5 = C1_P4 + PHASE_ENTRANCE_FRAMES + C1_SPEC_FLY
FRAMES_CYCLE_1_CONTENT = C1_P5
# Cycle 2
C2_P1 = PHASE_ENTRANCE_FRAMES + C2_INPUT_TYPING
C2_P2 = C2_P1 + PHASE_ENTRANCE_FRAMES + C2_INPUT_TO_DRAFT
C2_P3 = C2_P2 + PHASE_ENTRANCE_FRAMES + FRAMES_GENERATION_C2
C2_P4 = C2_P3 + PHASE_ENTRANCE_FRAMES + C2_BUFFER
C2_P5 = C2_P4 + PHASE_ENTRANCE_FRAMES + C2_SPEC_FLY
FRAMES_CYCLE_2_CONTENT = C2_P5
FRAMES_CYCLE_3_CONTENT = FRAMES_CYCLE_1_CONTENT
FRAMES_PAUSE = 15
FRAMES_CYCLE_1 = FRAMES_CYCLE_1_CONTENT + FRAMES_PAUSE
FRAMES_CYCLE_2 = FRAMES_CYCLE_2_CONTENT + FRAMES_PAUSE
FRAMES_CYCLE_3 = FRAMES_CYCLE_3_CONTENT + FRAMES_PAUSE
CUMULATIVE_FRAMES = [0, FRAMES_CYCLE_1, FRAMES_CYCLE_1 + FRAMES_CYCLE_2, FRAMES_CYCLE_1 + FRAMES_CYCLE_2 + FRAMES_CYCLE_3]
n_frames = CUMULATIVE_FRAMES[-1]

def _progress_after_hold(phase_frame, hold, content_frames):
    """phase 前 hold 帧 progress=0，之后 (phase_frame-hold)/content_frames 线性到 1."""
    if phase_frame < hold:
        return 0.0
    return min(1.0, (phase_frame - hold) / content_frames)

def get_cycle_phase(frame):
    """每 phase 先换字、缩放(PHASE_ENTRANCE 前段)、暂停(后段)、再按 progress 执行动画."""
    cycle = 1
    for c in range(2, 4):
        if frame >= CUMULATIVE_FRAMES[c - 1]:
            cycle = c
    phase_frame = frame - CUMULATIVE_FRAMES[cycle - 1]
    if cycle == 2:
        if phase_frame >= FRAMES_CYCLE_2_CONTENT:
            return cycle, 'cycle_pause', _progress_after_hold(phase_frame - FRAMES_CYCLE_2_CONTENT, 0, FRAMES_PAUSE)
        if phase_frame < C2_P1:
            return cycle, 'input_typing', _progress_after_hold(phase_frame, PHASE_ENTRANCE_FRAMES, C2_INPUT_TYPING)
        elif phase_frame < C2_P2:
            return cycle, 'input_to_draft', _progress_after_hold(phase_frame - C2_P1, PHASE_ENTRANCE_FRAMES, C2_INPUT_TO_DRAFT)
        elif phase_frame < C2_P3:
            return cycle, 'generation', _progress_after_hold(phase_frame - C2_P2, PHASE_ENTRANCE_FRAMES, FRAMES_GENERATION_C2)
        elif phase_frame < C2_P4:
            return cycle, 'buffer_trainer_spec', _progress_after_hold(phase_frame - C2_P3, PHASE_ENTRANCE_FRAMES, C2_BUFFER)
        else:
            return cycle, 'spec_fly', _progress_after_hold(phase_frame - C2_P4, PHASE_ENTRANCE_FRAMES, C2_SPEC_FLY)
    else:
        if phase_frame >= FRAMES_CYCLE_1_CONTENT:
            return cycle, 'cycle_pause', _progress_after_hold(phase_frame - FRAMES_CYCLE_1_CONTENT, 0, FRAMES_PAUSE)
        if phase_frame < C1_P1:
            return cycle, 'input_typing', _progress_after_hold(phase_frame, PHASE_ENTRANCE_FRAMES, C1_INPUT_TYPING)
        elif phase_frame < C1_P2:
            return cycle, 'input_to_draft', _progress_after_hold(phase_frame - C1_P1, PHASE_ENTRANCE_FRAMES, C1_INPUT_TO_DRAFT)
        elif phase_frame < C1_P3:
            return cycle, 'generation', _progress_after_hold(phase_frame - C1_P2, PHASE_ENTRANCE_FRAMES, C1_GENERATION)
        elif phase_frame < C1_P4:
            return cycle, 'buffer_trainer_spec', _progress_after_hold(phase_frame - C1_P3, PHASE_ENTRANCE_FRAMES, C1_BUFFER)
        else:
            return cycle, 'spec_fly', _progress_after_hold(phase_frame - C1_P4, PHASE_ENTRANCE_FRAMES, C1_SPEC_FLY)

def animate(frame):
    # Clear dynamic elements
    for elem in state.user_elements:
        try:
            elem.remove()
        except:
            pass
    state.user_elements = []
    
    for elem in state.response_elements:
        try:
            elem.remove()
        except:
            pass
    state.response_elements = []
    
    for dot in state.data_dots:
        try:
            dot.remove()
        except:
            pass
    state.data_dots = []

    for elem in state.spec_fly_elements:
        try:
            elem.remove()
        except:
            pass
    state.spec_fly_elements = []

    cycle, phase, progress = get_cycle_phase(frame)
    
    # Determine user type and happiness for this cycle
    # Spec evolves: Medical only -> Medical+Research (after cycle 2 training)
    # Use symbols: + for Medical (cross), * for Research (star)
    if cycle == 1:
        user_type = 'Nurse'
        is_happy = False
        spec_type = 'Assistant with Health Speculator'
        result = 'get response quickly🥰'
        show_research = False
    elif cycle == 2:
        user_type = 'student'
        is_happy = False
        spec_type = 'Assistant with Health Speculator'
        result = 'get response slowly😭'
        show_research = (phase == 'spec_fly' and progress > 0.85) or (phase == 'cycle_pause')
    else:  # cycle 3
        user_type = 'student'
        # Start sad, become happy as output appears
        is_happy = (phase == 'generation' and progress > 0.2) or phase in ('buffer_trainer_spec', 'spec_fly', 'cycle_pause')
        spec_type = 'Assistant with Health+Math Speculator'
        result = 'get response quickly🥰'
        show_research = True  # Already trained
    
    # Update cycle text
    cycle_text.set_text(f'Cycle {cycle}/3: {user_type.capitalize()} using {spec_type} {result}')
    
    # 当前 phase 起始帧（用于本 phase 内局部帧）
    phase_frame = frame - CUMULATIVE_FRAMES[cycle - 1]
    if cycle == 2:
        phase_starts = {'input_typing': 0, 'input_to_draft': C2_P1, 'generation': C2_P2, 'buffer_trainer_spec': C2_P3, 'spec_fly': C2_P4, 'cycle_pause': FRAMES_CYCLE_2_CONTENT}
    else:
        phase_starts = {'input_typing': 0, 'input_to_draft': C1_P1, 'generation': C1_P2, 'buffer_trainer_spec': C1_P3, 'spec_fly': C1_P4, 'cycle_pause': FRAMES_CYCLE_1_CONTENT}
    phase_start = phase_starts.get(phase, 0)
    local_phase_frame = phase_frame - phase_start
    # status_text / cycle_text 出场动画：每个 phase 前 TEXT_HOLD_FRAMES 帧内先放大再缩回
    if local_phase_frame < TEXT_HOLD_FRAMES:
        t = local_phase_frame / TEXT_HOLD_FRAMES
        if t < 0.5:
            scale = 1.0 + 0.5 * (t / 0.5)
        else:
            scale = 1.5 - 0.5 * ((t - 0.5) / 0.5)
        status_text.set_fontsize(STATUS_TEXT_BASE_FONTSIZE * scale)
        cycle_text.set_fontsize(CYCLE_TEXT_BASE_FONTSIZE * scale)
    else:
        status_text.set_fontsize(STATUS_TEXT_BASE_FONTSIZE)
        cycle_text.set_fontsize(CYCLE_TEXT_BASE_FONTSIZE)
    
    # Reset box alphas
    for i, (box, label) in enumerate(zip(draft_boxes, draft_labels)):
        box.set_alpha(0.3)
        label.set_alpha(0.3)
        # Reset icons
        med_circle, med_cross = draft_med_icons[i]
        res_circle, res_star = draft_res_icons[i]
        med_circle.set_alpha(0.5)
        med_cross.set_alpha(0.5)
        if show_research:
            res_circle.set_alpha(0.5)
            res_star.set_alpha(0.5)
        else:
            res_circle.set_alpha(0)
            res_star.set_alpha(0)
    target_box.set_alpha(0.3)
    target_label.set_alpha(0.3)
    trainer_box.set_alpha(0.3)
    trainer_label.set_alpha(0.3)
    spec_box.set_alpha(0.3)
    spec_label.set_alpha(0.3)
    data_buffer.set_alpha(0.5)
    
    # User position (moved up; input/output boxes below)
    user_x, user_y = USER_X, USER_Y
    inp, out = INPUT_OUTPUT_BY_CYCLE[cycle - 1]

    # Data Buffer 进度条：status1/2 半满，status3 半满→全满，status4 全满→1/4→1/2，status5 半满
    if phase == 'input_typing' or phase == 'input_to_draft':
        _fill_level = 0.5
    elif phase == 'generation':
        _fill_level = 0.5 + 0.5 * progress
    elif phase == 'buffer_trainer_spec':
        if progress <= 0.5:
            _fill_level = 1.0 - 1.5 * progress
        else:
            _fill_level = 0.25 + 0.5 * (progress - 0.5)
    else:
        _fill_level = 0.5
    data_buffer_fill.set_height(_data_buffer_inner_h_max * _fill_level)

    if phase == 'input_typing':
        # User input 逐字出现; 用户图像从左到右切入 (entrance_progress 0→1)
        n_in = max(1, int(progress * len(inp)))
        input_content.set_text('\n'.join(textwrap.wrap(inp[:n_in], width=INPUT_WRAP_WIDTH)))
        output_content.set_text('')
        entrance = progress  # 0→1 时图像从左滑入
        state.user_elements = draw_user(ax, user_x, user_y, user_type, is_happy, 1.0, cycle=cycle, entrance_progress=entrance)
        status_text.set_text('Status1: User input...')

    elif phase == 'input_to_draft':
        # Input 显示完整; 小球 from input 右缘 to draft 左缘
        input_content.set_text('\n'.join(textwrap.wrap(inp, width=INPUT_WRAP_WIDTH)))
        output_content.set_text('')
        state.user_elements = draw_user(ax, user_x, user_y, user_type, is_happy, 1.0, cycle=cycle)
        draft_boxes[0].set_alpha(0.9)
        draft_labels[0].set_alpha(1.0)
        draft_med_icons[0][0].set_alpha(0.95)
        draft_med_icons[0][1].set_alpha(0.95)
        if show_research:
            draft_res_icons[0][0].set_alpha(0.95)
            draft_res_icons[0][1].set_alpha(0.95)
        # 每次 3 个小球，到达后再发下一批（本 phase 单批，progress 即本批进度）
        # 小球从 input 右边框的垂直中点出发，到 Draft 中心
        input_right_center_y = INPUT_BOX_Y + INPUT_BOX_H / 2
        draw_dots_batch_along_path(ax, INPUT_BOX_LEFT + INPUT_BOX_W, input_right_center_y, DRAFT_X, DRAFT_CENTER_Y, progress, n_dots=3, color=COLOR_AURORA)
        status_text.set_text('Status2: Assistant get the input...')

    elif phase == 'generation':
        # Draft、target、data buffer 高亮; output 逐字; 全部用线性 progress，cycle2 靠更多帧数均匀更慢
        input_content.set_text('\n'.join(textwrap.wrap(inp, width=INPUT_WRAP_WIDTH)))
        if cycle == 2:
            content_frame = phase_frame - C2_P2 - PHASE_ENTRANCE_FRAMES
            total_out_frames = FRAMES_GENERATION_C2 + C2_BUFFER
        else:
            content_frame = phase_frame - C1_P2 - PHASE_ENTRANCE_FRAMES
            total_out_frames = C1_GENERATION + C1_BUFFER
        progress_out = min(1.0, max(0.0, content_frame / total_out_frames))
        n_out = max(0, int(progress_out * len(out)))
        output_content.set_text('\n'.join(textwrap.wrap(out[:n_out], width=WRAP_WIDTH)) if n_out else '')
        state.user_elements = draw_user(ax, user_x, user_y, user_type, is_happy, 1.0, cycle=cycle, entrance_progress=1.0)
        draft_boxes[0].set_alpha(0.9)
        draft_labels[0].set_alpha(1.0)
        draft_med_icons[0][0].set_alpha(0.95)
        draft_med_icons[0][1].set_alpha(0.95)
        if show_research:
            draft_res_icons[0][0].set_alpha(0.95)
            draft_res_icons[0][1].set_alpha(0.95)
        target_box.set_alpha(0.9)
        target_label.set_alpha(1.0)
        data_buffer.set_alpha(0.9)
        # 每批 3 个小球，到达后再发下一批（循环：lap 进度 0→1 为一批）
        t_loop = (progress * 1.5) % 1.0
        draw_dots_batch_along_path(ax, INFERENCE_CENTER_X, DRAFT_Y, INFERENCE_CENTER_X, TARGET_Y + MODEL_BOX_H, t_loop, n_dots=3, color=COLOR_DRAFT)
        t2 = (progress * 1.5 + 0.33) % 1.0
        draw_dots_batch_along_path(ax, INFERENCE_CENTER_X, TARGET_Y, DATA_BUFFER_LEFT + 0.4, DATA_BUFFER_CY, t2, n_dots=3, color=COLOR_TARGET)
        t3 = (progress * 1.5 + 0.66) % 1.0
        output_center_x = OUTPUT_BOX_LEFT + OUTPUT_BOX_W / 2
        output_center_y = OUTPUT_BOX_Y + OUTPUT_BOX_H / 2
        draw_dots_batch_along_path(ax, INFERENCE_CENTER_X, TARGET_CENTER_Y, output_center_x, output_center_y, t3, n_dots=3, color=COLOR_TARGET)
        status_text.set_text('Status3: Assistant generate the output and transfer data to the buffer...')

    elif phase == 'buffer_trainer_spec':
        # output 继续逐字（与 stage3 连续，在整个 stage3+stage4 内 0→1）
        input_content.set_text('\n'.join(textwrap.wrap(inp, width=INPUT_WRAP_WIDTH)))
        if cycle == 2:
            content_frame = FRAMES_GENERATION_C2 + (phase_frame - C2_P3 - PHASE_ENTRANCE_FRAMES)
            total_out_frames = FRAMES_GENERATION_C2 + C2_BUFFER
        else:
            content_frame = C1_GENERATION + (phase_frame - C1_P3 - PHASE_ENTRANCE_FRAMES)
            total_out_frames = C1_GENERATION + C1_BUFFER
        progress_out = min(1.0, max(0.0, content_frame / total_out_frames))
        n_out = max(0, int(progress_out * len(out)))
        output_content.set_text('\n'.join(textwrap.wrap(out[:n_out], width=WRAP_WIDTH)) if n_out else '')
        state.user_elements = draw_user(ax, user_x, user_y, user_type, is_happy, 1.0, cycle=cycle, entrance_progress=1.0)
        data_buffer.set_alpha(0.9)
        trainer_box.set_alpha(0.6 + 0.3 * np.sin(progress * 8 * np.pi))
        trainer_label.set_alpha(0.9)
        spec_box.set_alpha(0.5 + 0.4 * progress)
        spec_label.set_alpha(0.5 + 0.4 * progress)
        # 每批 3 个小球，到达后再发下一批（phase 内多批：progress*2 每 0.5 一批）
        batch_p = (progress * 2) % 1.0
        draw_dots_batch_along_path(ax, DATA_BUFFER_CX, DATA_BUFFER_CY, TRAIN_CENTER_X, TRAINER_CENTER_Y, batch_p, n_dots=3, color=COLOR_DATA)
        draw_dots_batch_along_path(ax, TRAIN_CENTER_X, TRAINER_Y + MODEL_BOX_H, TRAIN_CENTER_X, SPEC_Y, batch_p, n_dots=3, color=COLOR_AURORA)
        # Status3 的 draft→target→buffer & output 小球在 Status4 用浅色重放（与 Status3 相同路径与节奏）
        light_alpha = 0.5
        t_loop = (progress * 1.5) % 1.0
        draw_dots_batch_along_path(ax, INFERENCE_CENTER_X, DRAFT_Y, INFERENCE_CENTER_X, TARGET_Y + MODEL_BOX_H, t_loop, n_dots=3, color=COLOR_DRAFT_LIGHT, alpha=light_alpha)
        t2 = (progress * 1.5 + 0.33) % 1.0
        draw_dots_batch_along_path(ax, INFERENCE_CENTER_X, TARGET_Y, DATA_BUFFER_LEFT + 0.4, DATA_BUFFER_CY, t2, n_dots=3, color=COLOR_TARGET_LIGHT, alpha=light_alpha)
        t3 = (progress * 1.5 + 0.66) % 1.0
        output_center_x = OUTPUT_BOX_LEFT + OUTPUT_BOX_W / 2
        output_center_y = OUTPUT_BOX_Y + OUTPUT_BOX_H / 2
        draw_dots_batch_along_path(ax, INFERENCE_CENTER_X, TARGET_CENTER_Y, output_center_x, output_center_y, t3, n_dots=3, color=COLOR_TARGET_LIGHT, alpha=light_alpha)
        status_text.set_text('Status4: Data Buffer is full! Start asynchronous training...')

    elif phase == 'spec_fly':
        # New Spec flies from Training Server to cover Draft1 in Inference Server
        # Ease-out: fast start, soft landing (progress^0.8)
        t = progress ** 0.8
        fly_x = SPEC_FLY_START[0] + (SPEC_FLY_END[0] - SPEC_FLY_START[0]) * t
        fly_y = SPEC_FLY_START[1] + (SPEC_FLY_END[1] - SPEC_FLY_START[1]) * t

        # Flying copy of New Spec (same color as Draft1)
        fly_box = FancyBboxPatch((fly_x, fly_y), SPEC_BOX_W, SPEC_BOX_H,
                                  boxstyle="round,pad=0.05,rounding_size=0.15",
                                  facecolor=COLOR_DRAFT, edgecolor='#ea580c', linewidth=2,
                                  alpha=0.95, zorder=20)
        ax.add_patch(fly_box)
        state.spec_fly_elements.append(fly_box)
        fly_label = ax.text(fly_x + SPEC_BOX_W / 2, fly_y + SPEC_BOX_H / 2, 'New\nSpec',
                            fontsize=10, fontweight='bold', color='white', ha='center', va='center',
                            alpha=0.95, zorder=21)
        state.spec_fly_elements.append(fly_label)

        # Fade out original New Spec in Training Server as it "leaves"
        spec_box.set_alpha(0.3 + 0.4 * (1 - progress))
        spec_label.set_alpha(0.3 + 0.4 * (1 - progress))
        # Dim Draft1 slightly so flying spec stands out when it lands
        draft_boxes[0].set_alpha(0.4 + 0.3 * progress)
        draft_labels[0].set_alpha(0.4 + 0.3 * progress)

        # User still visible (faded)
        user_alpha = 0.4
        state.user_elements = draw_user(ax, user_x, user_y, user_type, is_happy, user_alpha, cycle=cycle, entrance_progress=1.0)

        status_text.set_text('Status5: Training complete! New drafter deploying to Inference Server...')

    elif phase == 'cycle_pause':
        # 每个 cycle 结束暂停约 1s：保持 spec_fly 结束画面（New Spec 已覆盖 Draft）
        fly_x, fly_y = SPEC_FLY_END[0], SPEC_FLY_END[1]
        fly_box = FancyBboxPatch((fly_x, fly_y), SPEC_BOX_W, SPEC_BOX_H,
                                  boxstyle="round,pad=0.05,rounding_size=0.15",
                                  facecolor=COLOR_DRAFT, edgecolor='#ea580c', linewidth=2,
                                  alpha=0.95, zorder=20)
        ax.add_patch(fly_box)
        state.spec_fly_elements.append(fly_box)
        fly_label = ax.text(fly_x + SPEC_BOX_W / 2, fly_y + SPEC_BOX_H / 2, 'New\nSpec',
                            fontsize=10, fontweight='bold', color='white', ha='center', va='center',
                            alpha=0.95, zorder=21)
        state.spec_fly_elements.append(fly_label)
        spec_box.set_alpha(0.3)
        spec_label.set_alpha(0.3)
        draft_boxes[0].set_alpha(0.7)
        draft_labels[0].set_alpha(0.7)
        if show_research:
            draft_res_icons[0][0].set_alpha(0.95)
            draft_res_icons[0][1].set_alpha(0.95)
        state.user_elements = draw_user(ax, user_x, user_y, user_type, is_happy, 0.4, cycle=cycle, entrance_progress=1.0)
        status_text.set_text('Next cycle...')

    else:
        status_text.set_text('Ready for next user...')
    
    return []

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=80, blit=False)

# Save as MP4 (recommended for web: smaller, hardware-accelerated, no stutter)
out_mp4 = "./aurora_ttt_animated.mp4"
out_gif = "./aurora_ttt_animated.gif"
print(f"Saving animation to MP4: {out_mp4}")
print("This may take a few minutes...")
mp4_ok = False
try:
    anim.save(out_mp4, writer='ffmpeg', fps=15, dpi=100,
              metadata={'artist': 'aurora_animated'}, bitrate=2500)
    print(f"Animation saved to: {out_mp4}")
    mp4_ok = True
except Exception as e:
    print(f"FFmpeg MP4 save failed: {e}")
    print("Install ffmpeg (e.g. apt install ffmpeg / brew install ffmpeg) for MP4 output.")

# Optionally also save as GIF: set AURORA_SAVE_GIF=1, or fallback when MP4 failed
save_gif = os.environ.get("AURORA_SAVE_GIF", "").lower() in ("1", "true", "yes") or not mp4_ok
if save_gif:
    print(f"Saving animation to GIF: {out_gif} ...")
    anim.save(out_gif, writer='pillow', fps=15)
    print(f"Animation saved to: {out_gif}")

plt.close()
