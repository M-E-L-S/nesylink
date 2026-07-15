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
EXIT_PUSH_STEPS = 30

MOVE_TO_DELTA = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}
DELTA_TO_MOVE = {delta: action for action, delta in MOVE_TO_DELTA.items()}

ROUTE = [
    ("open_chest", None),
    ("press_button", None),
    ("go_exit", "south"),
    ("open_chest", None),
    ("go_exit", "north"),
    ("go_exit", "east"),
    ("open_chest", None),
    ("go_exit", "west"),
    ("go_exit", "west"),
    ("open_chest", None),
]


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


def _direction_to_adjacent(source: tuple[int, int], target: tuple[int, int]) -> int | None:
    return DELTA_TO_MOVE.get((target[0] - source[0], target[1] - source[1]))


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


class Task5Policy:
    def __init__(self, model_path: str | None = None):
        self.predictor = PixelToStatePredictor(model_path or str(Path("models/pixel2state/best.pt")))
        self.reset()

    def reset(self, seed: int | None = None, task_id: str | None = None):
        del seed, task_id
        self.route_index = 0
        self.last_action = ACTION_RIGHT
        self.last_player_tile: tuple[int, int] | None = None
        self.turn_settle_steps = 0
        self.action_repeat = 4
        self.exit_push_action: int | None = None
        self.exit_push_steps = 0
        self.attack_cooldown = 0
        self.combat_attack_steps = 0

    def act(self, obs: np.ndarray, info: dict) -> int:
        del info
        state = self.predictor.predict_state(obs)
        player_tile = _as_xy_tuple(state.get("player_tile")) or self.last_player_tile
        if player_tile is None:
            return self.last_action

        walls = set(_filter_positions(state.get("walls_all", [])))
        traps = set(_filter_positions(state.get("traps_all", [])))
        npcs = set(_filter_positions(state.get("npcs_all", [])))
        monsters = _filter_positions(state.get("monsters_all", []))
        chests = _filter_positions(state.get("chests_all", []))
        buttons = _filter_positions(state.get("buttons_all", []))
        exits = _filter_positions(state.get("exits_all", []))

        previous_player_tile = self.last_player_tile
        blocked = walls | traps | npcs | set(monsters)
        route_kind = ROUTE[self.route_index][0] if self.route_index < len(ROUTE) else None

        action = self._maybe_combat(player_tile, monsters, blocked)
        if action is None:
            action = self._route_action(player_tile, chests, buttons, exits, blocked)
        if action is None:
            action = ACTION_WAIT

        if route_kind != "go_exit":
            action = self._apply_turn_settle(previous_player_tile, player_tile, int(action))

        self.last_player_tile = player_tile
        self.last_action = int(action)
        return int(action)

    def _maybe_combat(
        self,
        player_tile: tuple[int, int],
        monsters: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        if not monsters or self.route_index < 5:
            self.combat_attack_steps = 0
            return None
        if self.combat_attack_steps >= 6:
            return None

        monster = min(monsters, key=lambda pos: _manhattan(player_tile, pos))
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
            self.combat_attack_steps += 1
            return ACTION_A
        if _manhattan(player_tile, monster) == 1:
            facing_action = _direction_to_adjacent(player_tile, monster)
            if facing_action == self.last_action:
                self.attack_cooldown = 4
                self.combat_attack_steps += 1
                return ACTION_A
            return facing_action

        goals = {
            neighbor
            for neighbor in _neighbors(monster)
            if _inside(neighbor) and neighbor not in blocked
        }
        return _bfs_next_action(player_tile, goals, blocked - {monster})

    def _route_action(
        self,
        player_tile: tuple[int, int],
        chests: list[tuple[int, int]],
        buttons: list[tuple[int, int]],
        exits: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        if self.route_index >= len(ROUTE):
            return ACTION_WAIT

        kind, target = ROUTE[self.route_index]
        if kind == "open_chest":
            return self._open_visible_chest(player_tile, chests, blocked)
        if kind == "press_button":
            return self._press_visible_button(player_tile, buttons, blocked)
        if kind == "go_exit":
            return self._go_visible_exit(player_tile, exits, str(target), blocked)
        return None

    def _open_visible_chest(
        self,
        player_tile: tuple[int, int],
        chests: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        if not chests:
            self.route_index += 1
            return ACTION_WAIT

        chest = min(chests, key=lambda pos: _manhattan(player_tile, pos))
        if _manhattan(player_tile, chest) == 1:
            self.route_index += 1
            return ACTION_A

        goals = {
            neighbor
            for neighbor in _neighbors(chest)
            if _inside(neighbor) and neighbor not in blocked
        }
        return _bfs_next_action(player_tile, goals, blocked | {chest})

    def _press_visible_button(
        self,
        player_tile: tuple[int, int],
        buttons: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        if not buttons:
            self.route_index += 1
            return ACTION_WAIT

        button = min(buttons, key=lambda pos: _manhattan(player_tile, pos))
        if player_tile == button:
            self.route_index += 1
            return ACTION_WAIT
        return _bfs_next_action(player_tile, {button}, blocked - {button})

    def _go_visible_exit(
        self,
        player_tile: tuple[int, int],
        exits: list[tuple[int, int]],
        direction: str,
        blocked: set[tuple[int, int]],
    ) -> int | None:
        push_action = self._action_for_direction(direction)
        if self.exit_push_steps > 0:
            self.exit_push_steps -= 1
            if self.exit_push_steps == 0:
                self.route_index += 1
                self.exit_push_action = None
            return self.exit_push_action or push_action

        target_exits = self._exits_for_direction(exits, direction)
        if not target_exits:
            self.route_index += 1
            self.exit_push_steps = 0
            self.exit_push_action = None
            return ACTION_WAIT

        exit_tile = min(target_exits, key=lambda pos: _manhattan(player_tile, pos))
        if player_tile == exit_tile or self._near_exit_band(player_tile, exit_tile):
            self.exit_push_action = push_action
            self.exit_push_steps = self._ticks_to_actions(EXIT_PUSH_STEPS)
            return push_action
        return _bfs_next_action(player_tile, {exit_tile}, blocked)

    def _exits_for_direction(self, exits: list[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
        if direction == "north":
            return [tile for tile in exits if tile[1] == 0]
        if direction == "south":
            return [tile for tile in exits if tile[1] == GRID_HEIGHT - 1]
        if direction == "west":
            return [tile for tile in exits if tile[0] == 0]
        if direction == "east":
            return [tile for tile in exits if tile[0] == GRID_WIDTH - 1]
        return []

    def _near_exit_band(self, player_tile: tuple[int, int], exit_tile: tuple[int, int]) -> bool:
        if exit_tile[0] == 0:
            return player_tile[0] <= 1 and player_tile[1] == exit_tile[1]
        if exit_tile[0] == GRID_WIDTH - 1:
            return player_tile[0] >= GRID_WIDTH - 2 and player_tile[1] == exit_tile[1]
        if exit_tile[1] == 0:
            return player_tile[1] <= 1 and player_tile[0] == exit_tile[0]
        if exit_tile[1] == GRID_HEIGHT - 1:
            return player_tile[1] >= GRID_HEIGHT - 2 and player_tile[0] == exit_tile[0]
        return False

    def _action_for_direction(self, direction: str) -> int:
        if direction == "north":
            return ACTION_UP
        if direction == "south":
            return ACTION_DOWN
        if direction == "east":
            return ACTION_RIGHT
        return ACTION_LEFT

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

    def _ticks_to_actions(self, ticks: int) -> int:
        return max(1, int((ticks + self.action_repeat - 1) // self.action_repeat))


policy = Task5Policy()
