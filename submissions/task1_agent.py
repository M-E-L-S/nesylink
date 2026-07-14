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

MOVE_TO_DELTA = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}

DELTA_TO_MOVE = {delta: action for action, delta in MOVE_TO_DELTA.items()}

GRID_WIDTH = 10
GRID_HEIGHT = 8
TURN_SETTLE_STEPS = 8
EXIT_PUSH_STEPS = 24
TASK1_NORTH_EXIT_TILES = [(4, 0), (5, 0)]


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


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(tile: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = tile
    return [
        (x, y - 1),
        (x, y + 1),
        (x - 1, y),
        (x + 1, y),
    ]


def _inside(tile: tuple[int, int]) -> bool:
    return 0 <= tile[0] < GRID_WIDTH and 0 <= tile[1] < GRID_HEIGHT


def _is_border_tile(tile: tuple[int, int]) -> bool:
    x, y = tile
    return x == 0 or x == GRID_WIDTH - 1 or y == 0 or y == GRID_HEIGHT - 1


def _select_task1_exit(exits: list[tuple[int, int]]) -> list[tuple[int, int]]:
    top_exits = [exit_tile for exit_tile in exits if exit_tile[1] == 0]
    if top_exits:
        return top_exits
    return [exit_tile for exit_tile in exits if _is_border_tile(exit_tile)]


def _bfs_next_action(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> int | None:
    if start in goals:
        return ACTION_WAIT

    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    found_goal: tuple[int, int] | None = None

    while queue:
        current = queue.popleft()
        if current in goals:
            found_goal = current
            break
        for nxt in _neighbors(current):
            if not _inside(nxt) or nxt in blocked or nxt in parent:
                continue
            parent[nxt] = current
            queue.append(nxt)

    if found_goal is None:
        return None

    step = found_goal
    while parent[step] is not None and parent[step] != start:
        step = parent[step]

    first = step if parent[step] == start else found_goal
    dx = first[0] - start[0]
    dy = first[1] - start[1]
    return DELTA_TO_MOVE.get((dx, dy))


class Task1Policy:
    def __init__(self, model_path: str | None = None):
        resolved_model_path = model_path or str(Path("models/pixel2state/best.pt"))
        self.predictor = PixelToStatePredictor(resolved_model_path)
        self.reset()

    def reset(self, seed: int | None = None, task_id: str | None = None):
        del seed, task_id
        self.phase = "get_key"
        self.last_action = ACTION_RIGHT
        self.last_player_tile: tuple[int, int] | None = None
        self.turn_settle_steps = 0
        self.exit_push_steps = 0
        self.exit_align_action: int | None = None
        self.exit_align_steps = 0
        self.action_repeat = 4
        self.key_chest_tile: tuple[int, int] | None = None
        self.chest_opened = False
        self.dynamic_blocked: set[tuple[int, int]] = set()
        self.stuck_move_steps = 0
        self.stuck_action: int | None = None
        self.nudge_script: list[int] = []

    def act(self, obs: np.ndarray, info: dict) -> int:
        del info
        state = self.predictor.predict_state(obs)

        player_tile = _as_xy_tuple(state.get("player_tile"))
        if player_tile is None:
            player_tile = self.last_player_tile
        if player_tile is None:
            return self.last_action

        previous_player_tile = self.last_player_tile

        walls = set(_filter_positions(state.get("walls_all", [])))
        chests = _filter_positions(state.get("chests_all", []))
        exits = _filter_positions(state.get("exits_all", []))
        self._update_stuck_memory(player_tile)
        walls |= self.dynamic_blocked

        if self.phase == "get_key" and not chests:
            self.chest_opened = True
        self.phase = "go_exit" if self.chest_opened else "get_key"

        if self.phase == "get_key":
            action = self._act_get_key(player_tile, chests, walls)
        else:
            action = self._act_go_exit(player_tile, exits, walls)

        if action is None:
            action = self._greedy_fallback(player_tile, chests if self.phase == "get_key" else exits, walls)
        if action is None:
            action = ACTION_WAIT

        action = self._recover_if_stuck(player_tile, int(action))
        if not self._in_exit_open_loop(player_tile):
            action = self._apply_turn_settle(previous_player_tile, player_tile, int(action))
        self.last_player_tile = player_tile
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

    def _in_exit_open_loop(self, player_tile: tuple[int, int]) -> bool:
        return self.phase == "go_exit" and (
            player_tile[1] == 0
            or self.exit_align_steps > 0
            or self.exit_push_steps > 0
        )

    def _apply_turn_settle(
        self,
        previous_player_tile: tuple[int, int] | None,
        player_tile: tuple[int, int],
        desired_action: int,
    ) -> int:
        if previous_player_tile is None:
            return desired_action

        desired_is_move = desired_action in MOVE_TO_DELTA
        last_is_move = self.last_action in MOVE_TO_DELTA
        tile_changed = player_tile != previous_player_tile

        if tile_changed and desired_is_move and last_is_move and desired_action != self.last_action:
            self.turn_settle_steps = self._ticks_to_actions(TURN_SETTLE_STEPS)

        if (
            self.turn_settle_steps > 0
            and desired_is_move
            and last_is_move
            and desired_action != self.last_action
        ):
            self.turn_settle_steps -= 1
            return self.last_action

        self.turn_settle_steps = 0
        return desired_action

    def _act_get_key(
        self,
        player_tile: tuple[int, int],
        chests: list[tuple[int, int]],
        walls: set[tuple[int, int]],
    ) -> int | None:
        if not chests:
            return ACTION_WAIT

        target_chest = min(chests, key=lambda pos: _manhattan(player_tile, pos))
        self.key_chest_tile = target_chest
        if _manhattan(player_tile, target_chest) == 1:
            self.chest_opened = True
            return ACTION_A

        goals = {
            neighbor
            for neighbor in _neighbors(target_chest)
            if _inside(neighbor) and neighbor not in walls and neighbor != target_chest
        }
        if not goals:
            return None
        return _bfs_next_action(player_tile, goals, walls | {target_chest})

    def _act_go_exit(
        self,
        player_tile: tuple[int, int],
        exits: list[tuple[int, int]],
        walls: set[tuple[int, int]],
    ) -> int | None:
        if self.exit_push_steps > 0:
            self.exit_push_steps -= 1
            return ACTION_UP

        if self.exit_align_steps > 0 and self.exit_align_action is not None:
            self.exit_align_steps -= 1
            if self.exit_align_steps == 0:
                self.exit_push_steps = EXIT_PUSH_STEPS
            return self.exit_align_action

        if player_tile[1] == 0:
            if player_tile[0] < 4:
                self.exit_align_action = ACTION_RIGHT
                self.exit_align_steps = self._ticks_to_actions((4 - player_tile[0]) * 16)
                self.exit_align_steps -= 1
                if self.exit_align_steps == 0:
                    self.exit_push_steps = self._ticks_to_actions(EXIT_PUSH_STEPS)
                return self.exit_align_action
            if player_tile[0] > 5:
                self.exit_align_action = ACTION_LEFT
                self.exit_align_steps = self._ticks_to_actions((player_tile[0] - 5) * 16)
                self.exit_align_steps -= 1
                if self.exit_align_steps == 0:
                    self.exit_push_steps = self._ticks_to_actions(EXIT_PUSH_STEPS)
                return self.exit_align_action
            self.exit_align_action = None
            self.exit_align_steps = 0
            self.exit_push_steps = self._ticks_to_actions(EXIT_PUSH_STEPS)
            return ACTION_UP

        border_exits = _select_task1_exit(exits)
        if not border_exits:
            border_exits = TASK1_NORTH_EXIT_TILES
        if not border_exits:
            return ACTION_UP

        best_exit = min(border_exits, key=lambda pos: _manhattan(player_tile, pos))
        if player_tile == best_exit:
            return self._push_exit_direction(best_exit)

        action = _bfs_next_action(player_tile, {best_exit}, walls)
        if action is not None:
            return action

        goals = {
            neighbor
            for neighbor in _neighbors(best_exit)
            if _inside(neighbor) and neighbor not in walls
        }
        if goals:
            return _bfs_next_action(player_tile, goals, walls)
        return None

    def _push_exit_direction(self, exit_tile: tuple[int, int]) -> int:
        x, y = exit_tile
        if y == 0:
            return ACTION_UP
        if y == GRID_HEIGHT - 1:
            return ACTION_DOWN
        if x == 0:
            return ACTION_LEFT
        if x == GRID_WIDTH - 1:
            return ACTION_RIGHT
        return ACTION_UP

    def _ticks_to_actions(self, ticks: int) -> int:
        return max(1, int((ticks + self.action_repeat - 1) // self.action_repeat))

    def _greedy_fallback(
        self,
        player_tile: tuple[int, int],
        targets: list[tuple[int, int]],
        walls: set[tuple[int, int]],
    ) -> int | None:
        if not targets:
            return None

        target = min(targets, key=lambda pos: _manhattan(player_tile, pos))
        candidates: list[tuple[int, int, int]] = []
        for action, (dx, dy) in MOVE_TO_DELTA.items():
            nxt = (player_tile[0] + dx, player_tile[1] + dy)
            if not _inside(nxt) or nxt in walls:
                continue
            candidates.append((_manhattan(nxt, target), action, action))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]


policy = Task1Policy()
