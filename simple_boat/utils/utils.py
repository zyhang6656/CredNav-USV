# utils
import math
import random

import numpy as np
from matplotlib.patches import Polygon

# ‑‑‑ MPC -------------------------------------------------------------------

def f_dyn(x, dt):
    """
    动态障碍物状态转移函数
    x = [x, y, θ, ω]
    """
    x_, y_, theta, omega = x
    base_r = 0.05
    amp = 0.3 * np.sin(0.1 * dt)
    v = 0.05 + base_r * amp  # 模拟扰动速度
    x_new = x_ + dt * v * np.cos(theta)
    y_new = y_ + dt * v * np.sin(theta)
    theta_new = theta + dt * omega
    return np.array([x_new, y_new, theta_new, omega])

def h_dyn(x):
    """
    观测函数：仅观测位置
    """
    return x[:2]

'''
CV (constant-velocity)
'''
def f_cv(x, dt):
    """CV: x = [px, py, vx, vy]"""
    px, py, vx, vy = x
    return np.array([px + vx*dt,
                     py + vy*dt,
                     vx, vy])

def h_cv(x):
    """只量测位置"""
    return x[:2]

def normalize_angle_0_to_2pi(angle):
    '''
    把任意角度归一化到 [0, 2π] 区间
    '''
    return angle % (2 * np.pi)

def compute_angle_error(target_angle, current_angle):
    """计算从current到target最短旋转方向和大小，返回[-π, π]"""
    # 归一化到 [0, 2π]
    target_angle = target_angle % (2 * np.pi)
    current_angle = current_angle % (2 * np.pi)

    # 计算差值,顺时针为正方向
    error = target_angle - current_angle

    # wrap到 [-π, π]
    if error > np.pi:
        error -= 2 * np.pi
    elif error < -np.pi:
        error += 2 * np.pi
    return error

def angle_min_diff(target, current):
    """
        计算两个角度在圆周上的最近差值（绝对值最小的差）
        参数:
            target (float): 目标角度（0 ≤ target < 2π）
            current (float): 当前角度（0 ≤ current < 2π）

        返回:
            float: 在圆周上最近的绝对角度差（范围 [0, π]）
        """
    delta = abs(target - current)
    return delta if delta <= math.pi else 2 * math.pi - delta

def distance_to_goal(ship_state, goal):
    x, y = ship_state[0], ship_state[1]
    gx, gy = goal
    return math.hypot(x - gx, y - gy)

# def distance_to_nearest_obstacle(x, y, obstacle_coords):
#     """
#         返回当前位置 (x,y) 到最近障碍物的欧氏距离
#     """
#     if len(obstacle_coords) == 0:
#         return float('inf')
#     dists = np.linalg.norm(obstacle_coords - np.array([y, x]), axis=1)
#     return dists.min()

    # 静态和动态
def check_collision(x, y, grid_map, dyn_pos, dyn_radius, H, W):
    row, col = int(round(y)), int(round(x))
    # 静态
    if row < 0 or row >= H or col < 0 or col >= W:
        return True
    if grid_map[row, col] == 1:
        return True
    # 动态：圆形距离碰撞
    for pos in dyn_pos:
        if np.linalg.norm(pos - np.array([x, y])) <= dyn_radius + 0.3:
            return True
    return False

def in_bounds(x, y, W, H):
    """
    判断当前位置是否在地图边界内
    Returns:
        bool: 如果在边界内，返回 True，否则返回 False
    """
    return 0 <= x <= W and 0 <= y <= H
# 全局转局部
def world_to_local(x, y, usv_state):
    x0, y0, psi = usv_state[0], usv_state[1], usv_state[2]
    dx, dy = x - x0, y - y0
    lx = dx * np.cos(-psi) - dy * np.sin(-psi)
    ly = dx * np.sin(-psi) + dy * np.cos(-psi)
    return lx, ly
# 局部转全局
def local_to_world(points_local, usv_state):
    x0, y0, psi = usv_state[0], usv_state[1], usv_state[2]
    R = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi),  np.cos(psi)]
    ])
    return points_local @ R.T + np.array([x0, y0])
# =====================  LiDAR 降维到 15 维向量  =====================
def build_lidar_feat(lidar_scan: np.ndarray, lidar_range: float) -> np.ndarray:
    """
    将 N 束雷达射程压缩为 15 维特征：
    0-11 : 每 30° 扇区最近距离 / lidar_range
    12   : cos(theta_min) （最近障碍方向）
    13   : sin(theta_min)
    14   : d_min / lidar_range
        均归一化
    """
    N = len(lidar_scan)
    n_per_sector = max(1, N // 12)
    d_sector = []
    for s in range(12):
        seg = lidar_scan[s * n_per_sector:(s + 1) * n_per_sector]
        d_sector.append(seg.min() / lidar_range)
    idx_min = int(np.argmin(lidar_scan))
    theta_min = -np.pi + 2 * np.pi * idx_min / N
    cos_t, sin_t = np.cos(theta_min), np.sin(theta_min)
    d_min = lidar_scan[idx_min] / lidar_range
    return np.array(d_sector + [cos_t, sin_t, d_min], dtype=np.float32)   # shape (15,)
# 小船大致样子
def draw_usv(ax, x, y, psi, length=2.0, width=1.08, color='red'):
    """
    以 (x, y) 为中心、朝向为 psi 的船体图形绘制。
    船体近似为前尖后椭圆的结构。
    """
    # 船体局部坐标（前尖、左右两侧、后部略圆）
    shape_local = np.array([
        [length * 0.5, 0],  # 船头尖角
        [length * 0.15, -width * 0.5],  # 左前侧
        [-length * 0.4, -width * 0.5],  # 左后侧
        [-length * 0.5, -width * 0.3],  # 左后椭圆
        [-length * 0.5, width * 0.3],  # 右后椭圆
        [-length * 0.4, width * 0.5],  # 右后侧
        [length * 0.15, width * 0.5],  # 右前侧
    ])

    # 旋转 + 平移到世界坐标
    rot = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi), np.cos(psi)]
    ])
    shape_global = shape_local @ rot.T + np.array([x, y])

    # patch = Polygon(shape_global, closed=True, edgecolor='black', linewidth=0.3, alpha=0.9)
    patch = Polygon(shape_global, closed=True, facecolor=color, edgecolor='black', linewidth=0.3, alpha=0.9)
    ax.add_patch(patch)


def draw_obstacles(ax, x, y, psi, length=2.0, width=1.08, color=None):
    """
    以 (x, y) 为中心、朝向为 psi 的船体图形绘制。
    船体近似为前尖后椭圆的结构。
    """
    # 如果未指定颜色，随机生成除红色外的颜色
    if color is None:
        # 定义除红色外的常用颜色列表
        colors = ['blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta',
                  'lime', 'teal', 'navy', 'maroon', 'olive', 'brown', 'gray', 'black']
        color = random.choice(colors)

    # 船体局部坐标（前尖、左右两侧、后部略圆）
    shape_local = np.array([
        [length * 0.5, 0],  # 船头尖角
        [length * 0.15, -width * 0.5],  # 左前侧
        [-length * 0.4, -width * 0.5],  # 左后侧
        [-length * 0.5, -width * 0.3],  # 左后椭圆
        [-length * 0.5, width * 0.3],  # 右后椭圆
        [-length * 0.4, width * 0.5],  # 右后侧
        [length * 0.15, width * 0.5],  # 右前侧
    ])

    # 旋转 + 平移到世界坐标
    rot = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi), np.cos(psi)]
    ])
    shape_global = shape_local @ rot.T + np.array([x, y])

    # patch = Polygon(shape_global, closed=True, edgecolor='black', linewidth=0.3, alpha=0.9, color=color)
    patch = Polygon(shape_global, closed=True, facecolor=color, edgecolor='black', linewidth=0.3, alpha=0.9)
    ax.add_patch(patch)

