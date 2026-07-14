from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from experiments.pixel2state.infer import PixelToStatePredictor


ACTION_WAIT = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_A = 5

GRID_WIDTH = 10
GRID_HEIGHT = 8
TURN_SETTLE_STEPS = 8
EXIT_PUSH_STEPS = 24

MOVE_TO_DELTA = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}
DELTA_TO_MOVE = {delta: action for action, delta in MOVE_TO_DELTA.items()}


def _as_xy_tuple(value) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        x = int(value[0])
        y = int(value[1])
    except Exception:
        return None
    if x < 0 or y < 0:
        return None
    return (x, y)


def _filter_positions(positions) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    try:
        iterable = list(positions)
    except Exception:
        return result
    for pos in iterable:
        xy = _as_xy_tuple(pos)
        if xy is not None:
            result.append(xy)
    return result


def _inside(tile: tuple[int, int]) -> bool:
    return 0 <= tile[0] < GRID_WIDTH and 0 <= tile[1] < GRID_HEIGHT


def _neighbors(tile: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = tile
    return [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _bfs_next_action(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> int | None:
    if start in goals:
        return ACTION_WAIT

    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    found: tuple[int, int] | None = None
    while queue:
        current = queue.popleft()
        if current in goals:
            found = current
            break
        for nxt in _neighbors(current):
            if not _inside(nxt) or nxt in blocked or nxt in parent:
                continue
            parent[nxt] = current
            queue.append(nxt)

    if found is None:
        return None

    step = found
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    first = step if parent[step] == start else found
    return DELTA_TO_MOVE.get((first[0] - start[0], first[1] - start[1]))


def _direction_to_adjacent(source: tuple[int, int], target: tuple[int, int]) -> int | None:
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    return DELTA_TO_MOVE.get((dx, dy))


class Task2Policy:
    def __init__(self, model_path: str | None = None):
        self.predictor = PixelToStatePredictor(model_path or str(Path("models/pixel2state/best.pt")))
        self.reset()

    def reset(self, seed: int | None = None, task_id: str | None = None):
        del seed, task_id
        self.phase = "kill_monster"
        self.last_action = ACTION_LEFT
        self.facing_action = ACTION_LEFT
        self.last_player_tile: tuple[int, int] | None = None
        self.turn_settle_steps = 0
        self.exit_push_steps = 0
        self.exit_script: list[int] = []
        self.exit_lane_align_steps = 0
        self.attack_cooldown = 0
        self.action_repeat = 4
        self.monster_tile: tuple[int, int] | None = None
        self.chest_tile: tuple[int, int] | None = None
        self.exit_tiles: list[tuple[int, int]] = []
        self.monster_killed = False
        self.chest_opened = False
        self.monster_attack_steps = 0
        self.dynamic_blocked: set[tuple[int, int]] = set()
        self.stuck_move_steps = 0
        self.stuck_action: int | None = None
        self.nudge_script: list[int] = []

    def act(self, obs: np.ndarray, info: dict) -> int:
        del info
        state = self.predictor.predict_state(obs)

        player_tile = _as_xy_tuple(state.get("player_tile")) or self.last_player_tile
        if player_tile is None:
            return self.last_action

        previous_player_tile = self.last_player_tile
        walls = set(_filter_positions(state.get("walls_all", [])))
        traps = set(_filter_positions(state.get("traps_all", [])))
        monsters = [
            monster
            for monster in _filter_positions(state.get("monsters_all", []))
            if monster != player_tile
        ]
        chests = _filter_positions(state.get("chests_all", []))
        exits = _filter_positions(state.get("exits_all", []))
        self._update_stuck_memory(player_tile)
        walls |= self.dynamic_blocked

        if monsters:
            self.monster_tile = min(monsters, key=lambda pos: _manhattan(player_tile, pos))
        if chests:
            self.chest_tile = min(chests, key=lambda pos: _manhattan(player_tile, pos))
        if exits:
            self.exit_tiles = exits

        if self.phase == "kill_monster" and self.monster_attack_steps >= 6 and not monsters:
            self.monster_killed = True
        if self.phase == "get_key" and not chests:
            self.chest_opened = True

        previous_phase = self.phase
        if not self.monster_killed:
            self.phase = "kill_monster"
            action = self._act_kill_monster(player_tile, monsters, walls | traps)
        elif not self.chest_opened:
            self.phase = "get_key"
            action = self._act_get_key(player_tile, chests, walls | traps)
        else:
            self.phase = "go_exit"
            if previous_phase != "go_exit" and self.exit_lane_align_steps <= 0:
                self.exit_lane_align_steps = 2
            remembered_blockers = {self.chest_tile} if self.chest_tile is not None else set()
            action = self._act_go_exit(player_tile, walls | traps | remembered_blockers)

        if action is None:
            action = ACTION_WAIT
        action = self._recover_if_stuck(player_tile, int(action))
        if self.phase not in {"go_exit", "kill_monster"}:
            action = self._apply_turn_settle(previous_player_tile, player_tile, int(action))

        self.last_player_tile = player_tile
        if int(action) in MOVE_TO_DELTA:
            self.facing_action = int(action)
        self.last_action = int(action)
        return int(action)

    def _update_stuck_memory(self, player_tile: tuple[int, int]) -> None:
        if self.last_player_tile is None or self.last_action not in MOVE_TO_DELTA:
            self.stuck_move_steps = 0
            self.stuck_action = None
            return
        if player_tile == self.last_player_tile:
            if self.stuck_action == self.last_action:
                self.stuck_move_steps += 1
            else:
                self.stuck_action = self.last_action
                self.stuck_move_steps = 1
        else:
            self.stuck_move_steps = 0
            self.stuck_action = None
            self.nudge_script = []

        if self.stuck_move_steps >= 8:
            dx, dy = MOVE_TO_DELTA[self.last_action]
            blocked = (player_tile[0] + dx, player_tile[1] + dy)
            if _inside(blocked):
                self.dynamic_blocked.add(blocked)

    def _recover_if_stuck(self, player_tile: tuple[int, int], desired_action: int) -> int:
        del player_tile
        if self.nudge_script:
            return self.nudge_script.pop(0)
        if self.stuck_move_steps < 4 or desired_action not in MOVE_TO_DELTA:
            return desired_action
        if desired_action in {ACTION_UP, ACTION_DOWN}:
            self.nudge_script = [ACTION_LEFT]
        else:
            self.nudge_script = [ACTION_UP]
        self.stuck_move_steps = 0
        return self.nudge_script.pop(0)

    def _apply_turn_settle(
        self,
        previous_player_tile: tuple[int, int] | None,
        player_tile: tuple[int, int],
        desired_action: int,
    ) -> int:
        if previous_player_tile is None:
            return desired_action
        if desired_action not in MOVE_TO_DELTA or self.last_action not in MOVE_TO_DELTA:
            self.turn_settle_steps = 0
            return desired_action
        if player_tile != previous_player_tile and desired_action != self.last_action:
            self.turn_settle_steps = self._ticks_to_actions(TURN_SETTLE_STEPS)
        if self.turn_settle_steps > 0 and desired_action != self.last_action:
            self.turn_settle_steps -= 1
            return self.last_action
        self.turn_settle_steps = 0
        return desired_action

    def _act_kill_monster(
        self,
        player_tile: tuple[int, int],
        monsters: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        monster = self.monster_tile
        if monster is None:
            return ACTION_WAIT
        distance = _manhattan(player_tile, monster)

        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
            self.monster_attack_steps += 1
            if self.monster_attack_steps >= 6:
                self.monster_killed = True
            return ACTION_A

        if distance == 1:
            facing_action = _direction_to_adjacent(player_tile, monster)
            if facing_action == self.facing_action:
                self.attack_cooldown = 4
                self.monster_attack_steps += 1
                if self.monster_attack_steps >= 6 or not monsters:
                    self.monster_killed = True
                return ACTION_A
            return facing_action

        goals = {
            neighbor
            for neighbor in _neighbors(monster)
            if _inside(neighbor) and neighbor not in blocked and 1 <= neighbor[1] <= 6
        }
        return _bfs_next_action(player_tile, goals, blocked)

    def _act_get_key(
        self,
        player_tile: tuple[int, int],
        chests: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        if chests:
            chest = min(chests, key=lambda pos: _manhattan(player_tile, pos))
            self.chest_tile = chest
        else:
            chest = self.chest_tile
        if chest is None:
            return ACTION_WAIT

        if _manhattan(player_tile, chest) == 1:
            self.chest_opened = True
            return ACTION_A
        goals = {
            neighbor
            for neighbor in _neighbors(chest)
            if _inside(neighbor) and neighbor not in blocked and 1 <= neighbor[1] <= 6
        }
        return _bfs_next_action(player_tile, goals, blocked | {chest})

    def _act_go_exit(self, player_tile: tuple[int, int], blocked: set[tuple[int, int]]) -> int:
        exits = [tile for tile in self.exit_tiles if tile[0] in {0, GRID_WIDTH - 1} or tile[1] in {0, GRID_HEIGHT - 1}]
        if not exits:
            exits = [(0, 3), (0, 4)]
        target_exit = self._select_reachable_exit(player_tile, exits, blocked)
        push_action = self._exit_push_action(target_exit)

        if self.exit_push_steps > 0:
            self.exit_push_steps -= 1
            return push_action
        if self._near_exit_band(player_tile, target_exit):
            self.exit_push_steps = self._ticks_to_actions(EXIT_PUSH_STEPS)
            return push_action

        if target_exit[0] in {0, GRID_WIDTH - 1} and player_tile[1] == target_exit[1]:
            if self.exit_lane_align_steps > 0:
                self.exit_lane_align_steps -= 1
                return ACTION_UP if target_exit[1] <= 3 else ACTION_DOWN

        if player_tile == target_exit:
            self.exit_push_steps = self._ticks_to_actions(EXIT_PUSH_STEPS)
            return push_action

        stand_goals = self._exit_stand_goals(target_exit, blocked)
        action = _bfs_next_action(player_tile, stand_goals or {target_exit}, blocked)
        if action is not None:
            return action

        return push_action

    def _select_reachable_exit(
        self,
        player_tile: tuple[int, int],
        exits: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> tuple[int, int]:
        candidates: list[tuple[int, int, tuple[int, int]]] = []
        for exit_tile in exits:
            stand_goals = self._exit_stand_goals(exit_tile, blocked)
            if not stand_goals:
                continue
            distance = min(_manhattan(player_tile, goal) for goal in stand_goals)
            candidates.append((distance, _manhattan(player_tile, exit_tile), exit_tile))
        if candidates:
            candidates.sort()
            return candidates[0][2]
        return min(exits, key=lambda pos: _manhattan(player_tile, pos))

    def _exit_stand_goals(
        self,
        exit_tile: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        x, y = exit_tile
        if x == 0:
            goals = {(1, y)}
        elif x == GRID_WIDTH - 1:
            goals = {(GRID_WIDTH - 2, y)}
        elif y == 0:
            goals = {(x, 1)}
        elif y == GRID_HEIGHT - 1:
            goals = {(x, GRID_HEIGHT - 2)}
        else:
            goals = {exit_tile}
        return {goal for goal in goals if _inside(goal) and goal not in blocked}

    def _near_exit_band(self, player_tile: tuple[int, int], exit_tile: tuple[int, int]) -> bool:
        if exit_tile[0] == 0:
            return player_tile[0] <= 1 and abs(player_tile[1] - exit_tile[1]) <= 1
        if exit_tile[0] == GRID_WIDTH - 1:
            return player_tile[0] >= GRID_WIDTH - 2 and abs(player_tile[1] - exit_tile[1]) <= 1
        if exit_tile[1] == 0:
            return player_tile[1] <= 1 and abs(player_tile[0] - exit_tile[0]) <= 1
        if exit_tile[1] == GRID_HEIGHT - 1:
            return player_tile[1] >= GRID_HEIGHT - 2 and abs(player_tile[0] - exit_tile[0]) <= 1
        return False

    def _exit_push_action(self, exit_tile: tuple[int, int]) -> int:
        if exit_tile[0] == 0:
            return ACTION_LEFT
        if exit_tile[0] == GRID_WIDTH - 1:
            return ACTION_RIGHT
        if exit_tile[1] == 0:
            return ACTION_UP
        if exit_tile[1] == GRID_HEIGHT - 1:
            return ACTION_DOWN
        return ACTION_LEFT

    def _ticks_to_actions(self, ticks: int) -> int:
        return max(1, int((ticks + self.action_repeat - 1) // self.action_repeat))

policy = Task2Policy()
