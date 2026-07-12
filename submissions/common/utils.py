"""共享工具函数：BFS寻路、动作转换等"""

from collections import deque
from typing import Tuple, List, Optional, Set

# ============ NesyLink 动作空间 ============
# 0: WAIT, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT, 5: BUTTON_A, 6: BUTTON_B
ACTION_WAIT = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_INTERACT = 5      # 交互/攻击 (BUTTON_A)
ACTION_ATTACK = 5        # 攻击 (BUTTON_A)
ACTION_SHIELD = 6        # 盾牌 (BUTTON_B)

# 方向动作列表
DIRECTION_ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
DIRECTION_NAMES = ["up", "down", "left", "right"]

# 默认动作（当没有更好的选择时）
DEFAULT_ACTION = ACTION_WAIT


def bfs_path(
    start: Tuple[int, int],
    target: Tuple[int, int],
    grid: List[List[int]],
    obstacles: Set[Tuple[int, int]] = None,
    max_steps: int = 1000,
) -> List[Tuple[int, int]]:
    """BFS寻路，返回从start到target的路径（不包括起点）"""
    if start == target:
        return []

    obstacles = obstacles or set()
    height, width = len(grid), len(grid[0])

    # 墙的语义类别是1
    wall_obstacles = set()
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                wall_obstacles.add((x, y))

    all_obstacles = wall_obstacles | obstacles

    queue = deque([(start[0], start[1], [])])
    visited = {start}

    while queue:
        x, y, path = queue.popleft()

        if len(path) > max_steps:
            continue

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)

            if pos == target:
                return path + [pos]

            if (0 <= nx < width and 0 <= ny < height and
                pos not in visited and pos not in all_obstacles):
                visited.add(pos)
                queue.append((nx, ny, path + [pos]))

    return []


def path_to_actions(path: List[Tuple[int, int]]) -> List[int]:
    """将路径转换为动作序列"""
    if not path:
        return [ACTION_WAIT]

    actions = []
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]

        # 确定方向
        if x2 > x1:
            action = ACTION_RIGHT
        elif x2 < x1:
            action = ACTION_LEFT
        elif y2 > y1:
            action = ACTION_DOWN
        elif y2 < y1:
            action = ACTION_UP
        else:
            action = ACTION_WAIT


    return actions


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """曼哈顿距离"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def is_adjacent(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
    """判断两个格子是否相邻（四方向）"""
    return manhattan_distance(p1, p2) == 1


def get_direction_to_target(
    current: Tuple[int, int],
    target: Tuple[int, int]
) -> int:
    """获取从current到target的移动方向动作"""
    dx = target[0] - current[0]
    dy = target[1] - current[1]

    # 优先水平移动，再垂直移动
    if abs(dx) >= abs(dy):
        if dx > 0:
            return ACTION_RIGHT
        elif dx < 0:
            return ACTION_LEFT
        else:
            return ACTION_WAIT
    else:
        if dy > 0:
            return ACTION_DOWN
        elif dy < 0:
            return ACTION_UP
        else:
            return ACTION_WAIT


def get_action_name(action: int) -> str:
    """获取动作名称（调试用）"""
    names = {
        ACTION_WAIT: "WAIT",
        ACTION_UP: "UP",
        ACTION_DOWN: "DOWN",
        ACTION_LEFT: "LEFT",
        ACTION_RIGHT: "RIGHT",
        ACTION_INTERACT: "BUTTON_A",
        ACTION_SHIELD: "BUTTON_B",
    }
    return names.get(action, f"UNKNOWN({action})")