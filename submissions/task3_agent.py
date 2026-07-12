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
    Task 3: 三房间联动 - 完全基于 info 状态
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

        # 目标位置（使用 tile 坐标）
        self.monster_pos = None
        self.chest_pos = None
        self.exit_pos = None

        # 状态追踪
        self.monster_killed = False
        self.has_key = False
        self._step_count = 0
        self._last_action = ACTION_RIGHT
        self._info = None

    def act(self, obs, info) -> int:
        """主决策入口 - 从 info 获取所有状态"""
        self._step_count += 1
        self._info = info
        state = self.predictor.predict_state(obs)

        self.grid=state["grid"]
        player_pos = state["player_tile"]

        # ============ 1. 从 info 提取所有状态 ============

        # player_pos = None
        # if "agent" in info:
        #     agent = info["agent"]
        #     if "tile" in agent:
        #         pos = agent["tile"]
        #         if isinstance(pos, (list, tuple)) and len(pos) == 2:
        #             player_pos = (pos[0], pos[1])
        #
        if player_pos is None:
            return ACTION_RIGHT

        # 1.2 当前房间
        self.current_room = info.get("env", {}).get("room_id", "unknown")

        # 1.3 物品状态（keys 是数量）
        inventory = info.get("inventory", {})
        self.has_key = inventory.get("keys", 0) > 0

        # 1.4 实体统计
        entities = info.get("entities", {})
        monsters_remaining = entities.get("monsters_remaining", 0)
        chests_remaining = entities.get("chests_remaining", 0)

        # 1.5 游戏状态
        game = info.get("game", {})
        world_completed = game.get("world_completed", False)

        # 1.6 事件（用于检测交互是否成功）
        events = info.get("events", {})
        event_flags = events.get("flags", {})

        # 检测怪物是否被击杀
        self.monster_killed = monsters_remaining == 0

        # 检测是否拿到钥匙（通过事件或 inventory）
        if event_flags.get("key_collected", False):
            self.has_key = True
        if inventory.get("keys", 0) > 0:
            self.has_key = True

        # ============ 2. 获取目标位置 ============

        # 2.1 怪物位置
        self.monster_pos = state["monsters_tile"][0]  # 第一个怪物


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

        self._last_action = action
        return action


    def _update_phase(self,  info):
        """根据当前状态更新阶段"""

        if info["entities"]["monsters_remaining"] > 0:
            room = "monster_hall"
        elif info["entities"]["chests_remaining"] > 0:
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
            target = (0, y)
        elif target_room == "key_room":
            target = (0, y)
        else:
            target = (4, y)

        return self._move_towards(player_pos, target)

    def _kill_monster(self, player_pos) -> int:
        """击杀怪物"""
        if self.monster_killed:
            return ACTION_LEFT

        if self.monster_pos[0] < 0 or self.monster_pos[1] < 0:
            self.monster_pos = None

        if self.monster_pos is None:
            x, y = player_pos
            if x < 5:
                return ACTION_RIGHT
            else:
                return ACTION_LEFT

        if is_adjacent(player_pos, self.monster_pos):
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

        if is_adjacent(player_pos, self.chest_pos):
            return ACTION_INTERACT

        return self._move_towards(player_pos, self.chest_pos)

    def _go_to_exit(self, player_pos) -> int:
        """回出口离开"""
        if self.exit_pos is None:
            return ACTION_RIGHT

        if player_pos == self.exit_pos:
            return ACTION_INTERACT

        return self._move_towards(player_pos, self.exit_pos)

    def _move_towards(self, current, target) -> int:
        """直接向目标移动"""
        if target is None:
            return ACTION_RIGHT

        dx = target[0] - current[0]
        dy = target[1] - current[1]


        if dx > 0:
            return ACTION_RIGHT
        elif dx < 0:
            return ACTION_LEFT
        else:
            if self._last_action == ACTION_RIGHT:
                return ACTION_RIGHT
            elif self._last_action == ACTION_LEFT:
                return ACTION_LEFT
            else:
                if dy > 0:
                   return ACTION_DOWN
                elif dy < 0:
                   return ACTION_UP
                else:
                   if self._last_action == ACTION_DOWN:
                       return ACTION_DOWN
                   elif self._last_action == ACTION_UP:
                       return ACTION_UP
                   else:
                       return ACTION_WAIT




# ============ 评估脚本需要的接口 ============

policy = Task3Policy()