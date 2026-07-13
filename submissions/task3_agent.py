"""Task 3: 三房间联动 - 完全基于 info 状态（修复版）"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional, Set, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from submissions.common.utils import (
    ACTION_WAIT, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_ATTACK, ACTION_INTERACT, ACTION_SHIELD,
    manhattan_distance, is_adjacent
)

from experiments.pixel2state.infer import PixelToStatePredictor

class Task3Policy:
    """
    Task 3: 三房间联动
    """

    def __init__(self, model_path: str = None):
        self.predictor = PixelToStatePredictor("models/pixel2state/best.pt")
        self.reset()

    def reset(self, seed: int = None, task_id: str = None):
        """重置状态机"""
        self.phase = "go_monster_hall"
        self.current_room = None
        self.grid = None
        self.path = []
        self.steps_since_phase_change = 0
        self.max_steps_per_phase = 200
        self.direction = 0
        self.monsters_remaining = 0
        self.chests_remaining = 0
        self.door_locked = True  #有锁的门默认在右边

        # 目标位置（使用 tile 坐标）
        self.monster_pos = None
        self.chest_pos = None
        self.door_pos_left = [11,11]
        self.door_pos_right = [11,11]

        # 状态追踪
        self.door_step = 0
        self.monster_killed = False
        self.has_key = False
        self._step_count = 0
        self._last_action = ACTION_RIGHT
        self._info = None

    def act(self, obs, info) -> int:
        """主决策入口 - 从 info 获取所有状态"""
        self._step_count += 1
        state = self.predictor.predict_state(obs)

        self.grid=state["grid"]


        player_pos = state["player_tile"]
        player_pos = None
        if "agent" in info:
             agent = info["agent"]
             if "tile" in agent:
                 pos = agent["tile"]
                 if isinstance(pos, (list, tuple)) and len(pos) == 2:
                     player_pos = (pos[0], pos[1])

        if player_pos is None:
            return ACTION_RIGHT

        # 1.2 当前房间
        door_pos = state["doors_tile"]
        door_remaining = state["doors_remaining"]
        for n in range (door_remaining):
            if door_pos[n][0] == 0 and door_pos[n][1] < self.door_pos_left[1]:
                self.door_pos_left = door_pos[n]
            elif door_pos[n][0] == 9 and door_pos[n][1] < self.door_pos_right[1]:
                self.door_pos_right = door_pos[n]

        self.monsters_remaining = state["monsters_remaining"]
        self.chests_remaining = state["chests_remaining"]

        # 检测怪物是否被击杀
        self.monster_killed = self.monsters_remaining == 0


        # ============ 2. 获取目标位置 ============

        # 2.1 怪物位置
        self.monster_pos = state["monsters_tile"][0]  # 第一个怪物

        if self.monster_pos[0] < 0 or self.monster_pos[1] < 0:
            self.monster_pos = None

        # ============ 3. 更新阶段 ============
        self._update_phase(info)

        # ============ 4. 执行当前阶段 ============
        self.steps_since_phase_change += 1
        if self.steps_since_phase_change > self.max_steps_per_phase:
            self.path = []
            self.steps_since_phase_change = 0

        action = self._execute_phase(player_pos)

        # 安全检查
        if action is None or not isinstance(action, int):
            return ACTION_RIGHT
        if action < 0 or action > 6:
            return ACTION_RIGHT

        if action == ACTION_RIGHT:
            self.direction = 0
        if action == ACTION_LEFT:
            self.direction = 1
        if action == ACTION_UP:
            self.direction = 2
        if action == ACTION_DOWN:
            self.direction = 3
        self._last_action = action
        return action


    def _update_phase(self,  info):
        """根据当前状态更新阶段"""

        if self.monsters_remaining > 0:
            room = "monster_hall"
        elif self.chests_remaining > 0:
            room = "key_room"
        else:
            room = "start_room"
        self.current_room = room

        # 状态机转换
        if self.phase == "go_monster_hall":
            if self.current_room == "monster_hall":
                self.phase = "kill_monster"
                self.steps_since_phase_change = 0

        elif self.phase == "kill_monster":
            if self.monster_killed:
                self.phase = "go_key_room"
                self.steps_since_phase_change = 0

        elif self.phase == "go_key_room":
            if self.current_room == "key_room":
                self.phase = "open_chest"
                self.steps_since_phase_change = 0

        elif self.phase == "open_chest":
            if self.has_key:
                self.phase = "go_exit"
                self.steps_since_phase_change = 0

        elif self.phase == "go_exit":
            if self.current_room == "start_room":
                pass

    def _execute_phase(self, player_pos) -> int:
        """执行当前阶段"""
        if self.phase == "go_monster_hall":
            return self._move_to_room(player_pos, "monster_hall")
        elif self.phase == "kill_monster":
            return self._kill_monster(player_pos)
        elif self.phase == "go_key_room":
            return self._move_to_room(player_pos, "key_room")
        elif self.phase == "open_chest":
            return self._open_chest(player_pos)
        elif self.phase == "go_exit":
            return self._go_to_exit(player_pos)
        else:
            return ACTION_RIGHT

    def _move_to_room(self, player_pos, target_room) -> int:
        """移动到目标房间"""
        if self.current_room == target_room:
            return ACTION_RIGHT

        x, y = player_pos

        if target_room == "monster_hall":
            target = self.door_pos_left
            if self.door_step>20:
                self.door_locked = False
                target = self.door_pos_right
        elif target_room == "key_room":
            if self.door_locked:
              target = self.door_pos_left
            else:
              target = self.door_pos_right
        else:
            target = (4, y)

        return self._move_towards(player_pos, target, True)

    def _kill_monster(self, player_pos) -> int:
        """击杀怪物"""
        if self.monster_killed:
            return ACTION_LEFT

        if self.monster_pos[0] < 0 or self.monster_pos[1] < 0:
            self.monster_pos = None

        if self.monster_pos is None:
                return ACTION_LEFT
        if is_adjacent(player_pos, self.monster_pos):
            if self.direction == 1 and self.monster_pos[0] < player_pos[0]:
               return ACTION_ATTACK
            elif self.direction == 3 and self.monster_pos[1] > player_pos[1]:
                return ACTION_ATTACK
            elif self._last_action == 0 and self.monster_pos[0] > player_pos[0]:
                return ACTION_ATTACK
            elif self._last_action == 2 and self.monster_pos[1] < player_pos[1]:
                return ACTION_ATTACK

        return self._move_towards(player_pos, self.monster_pos)

    def _open_chest(self, player_pos) -> int:
        """开宝箱拿钥匙"""
        if self.has_key:
            return ACTION_RIGHT
        chest_positions = []
        for y in range(8):
            for x in range(10):
                if self.grid[y][x] == 4:  # 钥匙
                    chest_positions.append((x, y))

        self.chest_pos = chest_positions[0]
        if self.chest_pos is None:
            x, y = player_pos
            if x < 8:
                return ACTION_RIGHT
            else:
                return ACTION_UP

        if manhattan_distance(player_pos, self.chest_pos) == 1:
            self.has_key = True
            return ACTION_INTERACT

        return self._move_towards(player_pos, self.chest_pos)

    def _go_to_exit(self, player_pos) -> int:
        """回出口离开"""
        if self.door_pos_right is None:
            return ACTION_RIGHT
        if self.door_locked:
           return self._move_towards(player_pos, self.door_pos_right)
        else:
           return self._move_towards(player_pos, self.door_pos_left)

    def _move_towards(self, current, target, door=False) -> int:
        """直接向目标移动"""
        if target is None:
            return ACTION_RIGHT

        dx = target[0] - current[0]
        dy = target[1] - current[1]

        if dx>=dy:
            if dx > 0:
                return ACTION_RIGHT
            elif dx < 0:
                return ACTION_LEFT
            else:
                if dy > 0:
                    return ACTION_DOWN
                elif dy < 0:
                    return ACTION_UP
                else:
                    if door:
                        return ACTION_LEFT
                    else:
                        return ACTION_RIGHT
        else:
            if dy > 0:
                return ACTION_DOWN
            elif dy < 0:
                return ACTION_UP
            else:
                if dx > 0:
                    return ACTION_RIGHT
                elif dx < 0:
                    return ACTION_LEFT
                else:
                    if door:
                        self.door_step+=1
                        return ACTION_LEFT
                    else:
                        return ACTION_RIGHT




# ============ 评估脚本需要的接口 ============

policy = Task3Policy()