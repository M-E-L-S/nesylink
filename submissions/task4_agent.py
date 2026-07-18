"""Task 4  """

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

class Task4Policy:


    def __init__(self, model_path: str = None):
        self.predictor = PixelToStatePredictor("models/pixel2state/best.pt")
        self.reset()

    def reset(self, seed: int = None, task_id: str = None):
        """重置状态机"""
        self.phase = "go_key_room"
        self.current_room = None
        self.grid = None
        self.path = []
        self.steps_since_phase_change = 0
        self.max_steps_per_phase = 200
        self.direction = 0
        self.monsters_remaining = 0
        self.chests_remaining = 0
        self.switches_remaining = 0
        self.is_bridge_room = True

        # 目标位置（使用 tile 坐标）
        self.monster_pos = None
        self.chest_pos = None
        self.switches_pos = None
        self.bridge_pos = None
        self.door_pos_left = [11, 11]
        self.door_pos_right = [11, 11]

        # 状态追踪
        self.room_change = True
        self.buffer_step = 0
        self.monster_killed = False
        self.has_key = False
        self.has_item = False
        self.switches_state = 0
        self.switches_size = 3
        self._step_count = 0
        self._last_player_pos = [7,4]
        self._last_action = ACTION_RIGHT
        self._info = None

    def act(self, obs, info) -> int:
        """主决策入口 - 从 info 获取所有状态"""
        self._step_count += 1
        state = self.predictor.predict_state(obs)

        self.grid = state["grid"]
        player_pos = None
        if "agent" in info:
            agent = info["agent"]
            if "tile" in agent:
                pos = agent["tile"]
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    player_pos = (pos[0], pos[1])

        player_pos = state["player_tile"]

        if player_pos is None:
            return ACTION_RIGHT

        # 1.2 当前房间
        self.bridge_pos = state["bridges_tile"]
        door_pos = state["doors_tile"]
        valid_doors = self._valid_tiles(door_pos)

        found_left = False
        found_right = False

        # 保留原始逻辑：优先找 x == 0 的左门、x == 9 的右门。
        # 但不再使用 door_remaining 索引，避免 color/spatial 下数组长度不一致崩溃。
        self.door_pos_right=self.door_pos_left=[11,11]
        for door in valid_doors:
            if door[0] == 0 and door[1] < self.door_pos_left[1]:
                self.door_pos_left = [door[0], door[1]]
                found_left = True
            elif door[0] == 9 and door[1] < self.door_pos_right[1]:
                self.door_pos_right = [door[0], door[1]]
                found_right = True
        # spatial 地图里门的 x 不一定刚好是 0/9。
        # 只有在原逻辑没找到门时，才用排序兜底，尽量不影响 original。
        if valid_doors:
            ordered_doors = sorted(valid_doors, key=lambda p: (p[0], p[1]))
            if not found_left and self.door_pos_left == [11, 11]:
                self.door_pos_left = [ordered_doors[0][0], ordered_doors[0][1]]
            if not found_right and self.door_pos_right == [11, 11]:
                self.door_pos_right = [ordered_doors[-1][0], ordered_doors[-1][1]]
        self.monsters_remaining = self._safe_int(state.get("monsters_remaining", 0), 0)
        self.chests_remaining = self._safe_int(state.get("chests_remaining", 0), 0)
        # 检测怪物是否被击杀
        if self.monsters_remaining == 0 and self.is_bridge_room and self.phase == "kill_monster":
            self.monster_killed = True

        # ============ 2. 获取目标位置 ============

        # 2.1 怪物位置
        self.bridge_pos = self._valid_tiles(self.bridge_pos)
        if len(self.bridge_pos) > 1:
            self.is_bridge_room = True
        else :
            self.is_bridge_room = False
        self.monster_pos = self._first_valid_tile(state.get("monsters_tile", []))
        self.chest_pos = self._first_valid_tile(state.get("chests_tile", []))
        self.switches_pos = self._first_valid_tile(state.get("switches_tile", []))
        if self.switches_pos is None:
            self.switches_remaining = 0
        else:
            self.switches_remaining = 1

        # ============ 3. 更新阶段 ============
        self._update_phase(info, player_pos)

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
        self._last_player_pos = player_pos
        return action

    def _valid_tiles(self, tiles):
        """从 predictor 输出中提取有效 tile，避免空数组/无效坐标导致崩溃。"""
        result = []
        if tiles is None:
            return result

        try:
            iterator = list(tiles)
        except TypeError:
            return result

        for tile in iterator:
            try:
                if len(tile) >= 2:
                    x = int(tile[0])
                    y = int(tile[1])
                    if x >= 0 and y >= 0:
                        result.append((x, y))
            except Exception:
                continue

        return result

    def _first_valid_tile(self, tiles):
        valid_tiles = self._valid_tiles(tiles)
        if not valid_tiles:
            return None
        return valid_tiles[0]

    def _safe_int(self, value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    def _update_phase(self, info,player_pos):
        """根据当前状态更新阶段"""
        if manhattan_distance(player_pos, self._last_player_pos) >4:
            self.room_change = True
        room = self.current_room
        if self.room_change:
            if self.monsters_remaining > 0:
                room = "monster_hall"
            #monster识别有误
            if self.phase=="go_monster_hall" and self.current_room == "bridge" and self.room_change:
                room = "monster_hall"
            elif self.chests_remaining > 0 and self.has_key ==False:
                room = "key_room"
            elif self.is_bridge_room:
                room = "bridge"
            elif self.chests_remaining > 0 and self.has_key:
                room = "item_room"
            elif self.switches_remaining > 0:
                room = "start_room"
            else:
                pass
        self.current_room = room
        self.room_change = False

        # 状态机转换
        if self.phase == "go_key_room":
            if self.current_room == "key_room":
                self.phase = "open_chest"
                self.steps_since_phase_change = 0

        elif self.phase == "open_chest":
            if self.has_key:
                self.phase = "switch_bridge_1"
                self.steps_since_phase_change = 0

        elif self.phase == "switch_bridge_1":
            if self.switches_state == 1:
                self.phase = "go_item_room"
                self.steps_since_phase_change = 0

        elif self.phase == "go_item_room":
            if self.current_room == "item_room":
                self.phase = "collect_item"
                self.steps_since_phase_change = 0

        elif self.phase == "collect_item":
            if self.has_item:
                self.phase = "switch_bridge_2"
                self.steps_since_phase_change = 0

        elif self.phase == "switch_bridge_2":
            if self.switches_state == 2:
                self.phase = "go_monster_hall"
                self.steps_since_phase_change = 0

        elif self.phase == "go_monster_hall":
            if self.current_room == "monster_hall":
                self.phase = "kill_monster"
                self.steps_since_phase_change = 0

        elif self.phase == "kill_monster":
            if self.monster_killed :
                self.phase = "go_bridge"
                self.steps_since_phase_change = 0

        elif self.phase == "go_bridge":
            if self.is_bridge_room:
                self.phase = "open_victory_chest"
                self.steps_since_phase_change = 0

        elif self.phase == "open_victory_chest":
            if self.is_bridge_room:
                pass

    def _execute_phase(self, player_pos) -> int:
        """执行当前阶段"""
        if self.phase == "go_key_room":
            return self._move_to_room(player_pos, "key_room")
        elif self.phase == "open_chest":
            return self._open_chest(player_pos)
        elif self.phase == "switch_bridge_1":
            return self._switch_bridge(player_pos)
        elif self.phase == "go_item_room":
            return self._move_to_room(player_pos, "item_room")
        elif self.phase == "collect_item":
            return self._open_chest(player_pos)
        elif self.phase == "switch_bridge_2":
            return self._switch_bridge(player_pos)
        elif self.phase == "go_monster_hall":
            return self._move_to_room(player_pos, "monster_hall")
        elif self.phase == "kill_monster":
            return self._kill_monster(player_pos)
        elif self.phase == "go_bridge":
            return self._go_to_exit(player_pos)
        elif self.phase == "open_victory_chest":
            return self._open_chest(player_pos)
        else:
            return ACTION_RIGHT

    def _move_to_room(self, player_pos, target_room) -> int:
        """移动到目标房间"""
        if self.current_room == target_room:
            return ACTION_RIGHT

        x, y = player_pos

        if target_room == "key_room":
            target = self.door_pos_right
        elif target_room == "monster_hall":
            target = self.door_pos_right
        elif target_room == "item_room":
                target = self.door_pos_right
                #桥上右门识别有误
                if self.door_pos_right[0] == self.door_pos_left[0] and self.current_room == "bridge":
                    return ACTION_RIGHT
        elif target_room == "bridge":
            target = self.door_pos_left
            if self.door_pos_right[0] == self.door_pos_left[0] and self.switches_state == 1:
                return ACTION_LEFT
        else:
            target = (4, y)

        return self._move_towards(player_pos, target, True)

    def _kill_monster(self, player_pos) -> int:
        """击杀怪物"""
        bridge_tile=self._first_valid_tile(self.bridge_pos)
        if self.monster_killed:
            return ACTION_UP
        if self.monster_pos is None:
            #monster识别有误
            if bridge_tile is None:
                return ACTION_UP
            if is_adjacent(player_pos, bridge_tile):
                if self.direction == 1 and bridge_tile[0] < player_pos[0]:
                    return ACTION_ATTACK
                elif self.direction == 3 and bridge_tile[1] > player_pos[1]:
                    return ACTION_ATTACK
                elif self.direction == 0 and bridge_tile[0] > player_pos[0]:
                    return ACTION_ATTACK
                elif self.direction == 2 and bridge_tile[1] < player_pos[1]:
                    return ACTION_ATTACK
            return self._move_towards(player_pos, bridge_tile)
        else:
            if is_adjacent(player_pos, self.monster_pos):
                if self.direction == 1 and self.monster_pos[0] < player_pos[0]:
                    return ACTION_ATTACK
                elif self.direction == 3 and self.monster_pos[1] > player_pos[1]:
                    return ACTION_ATTACK
                elif self.direction == 0 and self.monster_pos[0] > player_pos[0]:
                    return ACTION_ATTACK
                elif self.direction == 2 and self.monster_pos[1] < player_pos[1]:
                    return ACTION_ATTACK
            return self._move_towards(player_pos, self.monster_pos)



    def _open_chest(self, player_pos) -> int:
        """开宝箱拿钥匙"""
        if self.chest_pos is None:
            while self.buffer_step <= 10:
                self.buffer_step += 1
                return ACTION_UP
            return ACTION_INTERACT
        if manhattan_distance(player_pos, self.chest_pos) == 1:
            #视觉系统的识别位置可能不精确，先进行一些缓冲动作
            while self.buffer_step<=10:
                self.buffer_step += 1
                return self._move_towards(player_pos, self.chest_pos)
            self.has_key = True
            if self.phase == "collect_item":
                self.has_item = True
            self.buffer_step = 0
            return ACTION_INTERACT
        return self._move_towards(player_pos, self.chest_pos)


    def _switch_bridge(self, player_pos) -> int:
        if self.current_room != "bridge" and self.current_room != "start_room":
            return self._move_to_room(player_pos, "bridge")
        elif self.current_room == "bridge":
            return self._move_towards(player_pos, self.door_pos_left, True)
        else:
            if manhattan_distance(player_pos, self.switches_pos) == 1:
                self.switches_state=(self.switches_state + 1) % self.switches_size
                return ACTION_INTERACT
            return self._move_towards(player_pos, self.switches_pos)


    def _go_to_exit(self, player_pos) -> int:
        """回出口离开"""
        return self._move_towards(player_pos, self.door_pos_left)

    def _move_towards(self, current, target, door=False) -> int:
        """直接向目标移动"""
        if target is None:
            return ACTION_RIGHT
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        choice = 1
        if self.current_room == "bridge":
            if target == self.door_pos_left:
                choice = 0
            elif target == self.door_pos_right:
                choice = 1
        else:
            if abs(dx) >= abs(dy):
                choice = 1
            else:
                choice = 0
        if choice:
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
                    if door and self.current_room == "bridge":
                        if self.switches_state == 0:
                            return ACTION_UP
                        elif self.switches_state == 1:
                            return ACTION_RIGHT
                        else:
                            return ACTION_DOWN
                    elif door:
                        if self.current_room == "key_room":
                            return ACTION_DOWN
                        elif self.current_room == "monster_hall":
                            return ACTION_UP
                        else:
                            return ACTION_RIGHT
                    else:
                        return ACTION_LEFT
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
                        if self.current_room == "key_room":
                            return ACTION_DOWN
                        elif self.current_room == "monster_hall":
                            return ACTION_UP
                        else:
                            return ACTION_LEFT
                    else:
                        return ACTION_RIGHT

# ============ 评估脚本需要的接口 ============

policy = Task4Policy()