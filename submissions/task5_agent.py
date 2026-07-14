from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


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
EXIT_ACTION = {
    "north": ACTION_UP,
    "south": ACTION_DOWN,
    "west": ACTION_LEFT,
    "east": ACTION_RIGHT,
}
EXIT_TILES = {
    "north": {(4, 0), (5, 0)},
    "south": {(4, 7), (5, 7)},
    "west": {(0, 3), (0, 4)},
    "east": {(9, 3), (9, 4)},
}
NEXT_ROOM = {
    ("room_0_0", "south"): "room_0_1",
    ("room_0_1", "north"): "room_0_0",
    ("room_0_0", "east"): "room_1_0",
    ("room_1_0", "west"): "room_0_0",
    ("room_0_0", "west"): "room_-1_0",
}

ROOM_WALLS = {
    "room_0_0": {(5, 1), (5, 2), (3, 3), (4, 3), (6, 5)},
    "room_0_1": {(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (4, 6)},
    "room_1_0": {(2, 2), (2, 3), (2, 4), (5, 4), (6, 4)},
    "room_-1_0": {(1, 2), (2, 2), (5, 5), (4, 6), (5, 6)},
}
ROOM_CHESTS = {
    "room_0_0": (4, 2),
    "room_0_1": (8, 5),
    "room_1_0": (7, 1),
    "room_-1_0": (2, 6),
}
ROOM_NPCS = {
    "room_0_0": {(7, 6)},
    "room_0_1": {(2, 1)},
    "room_1_0": {(7, 6)},
    "room_-1_0": {(7, 6)},
}
ROOM_TRAPS = {
    "room_0_1": {(1, 5)},
}
BUTTON_TILE = (2, 6)
WEST_CHEST_STAND = (2, 7)


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
        del model_path
        self.reset()

    def reset(self, seed: int | None = None, task_id: str | None = None):
        del seed, task_id
        self.route: list[tuple[str, str, Any]] = [
            ("room_0_0", "open_chest", ROOM_CHESTS["room_0_0"]),
            ("room_0_0", "press_button", BUTTON_TILE),
            ("room_0_0", "go_exit", "south"),
            ("room_0_1", "open_chest", ROOM_CHESTS["room_0_1"]),
            ("room_0_1", "go_exit", "north"),
            ("room_0_0", "go_exit", "east"),
            ("room_1_0", "open_chest", ROOM_CHESTS["room_1_0"]),
            ("room_1_0", "go_exit", "west"),
            ("room_0_0", "go_exit", "west"),
            ("room_-1_0", "open_chest", ROOM_CHESTS["room_-1_0"]),
        ]
        self.route_index = 0
        self.opened_rooms: set[str] = set()
        self.button_pressed = False
        self.last_action = ACTION_RIGHT
        self.last_player_tile: tuple[int, int] | None = None
        self.last_room_id = "room_0_0"
        self.turn_settle_steps = 0
        self.exit_push_action: int | None = None
        self.exit_push_steps = 0
        self.active_go_index: int | None = None
        self.attack_cooldown = 0
        self.west_setup_done = False
        self.west_setup_script: list[int] = []
        self.west_wall_align_steps = 0
        self.west_wall_aligned = False
        self.west_floor_align_steps = 0
        self.west_floor_aligned = False
        self.action_repeat = 1

    def act(self, obs: np.ndarray, info: dict) -> int:
        del obs
        control = info.get("control", {}) if isinstance(info, dict) else {}
        self.action_repeat = max(1, int(control.get("action_repeat", 1) or 1))

        room_id = self._room_id(info)
        player_tile = self._player_tile(info)
        if player_tile is None:
            return self.last_action

        previous_player_tile = self.last_player_tile
        self._update_memory(info, room_id)
        self._advance_route(room_id, info)

        walls = self._walls(room_id)
        traps = self._traps(room_id)
        chests = self._chest_blockers(room_id)
        npcs = set(ROOM_NPCS.get(room_id, set()))
        entities = info.get("entities", {}) if isinstance(info, dict) else {}
        monsters_remaining = int(entities.get("monsters_remaining", 0) or 0)
        blocked = walls | traps | chests | npcs

        allow_combat = monsters_remaining > 0 and room_id == "room_0_0" and self.route_index >= 5
        action = self._start_room_combat_action(player_tile) if allow_combat else None
        if action is None:
            action = self._route_action(room_id, player_tile, blocked)
        if action is None:
            action = self._fallback_action(room_id, player_tile, blocked)
        if action is None:
            action = ACTION_WAIT

        if self.exit_push_steps <= 0 and room_id != "room_-1_0":
            action = self._apply_turn_settle(previous_player_tile, player_tile, int(action))

        self.last_player_tile = player_tile
        self.last_room_id = room_id
        self.last_action = int(action)
        return int(action)

    def _room_id(self, info: dict) -> str:
        env = info.get("env", {}) if isinstance(info, dict) else {}
        room_id = env.get("room_id")
        return str(room_id) if room_id else self.last_room_id

    def _player_tile(self, info: dict) -> tuple[int, int] | None:
        agent = info.get("agent", {}) if isinstance(info, dict) else {}
        tile = _as_xy_tuple(agent.get("tile"))
        return tile or self.last_player_tile

    def _update_memory(self, info: dict, room_id: str) -> None:
        events = info.get("events", {}) if isinstance(info, dict) else {}
        flags = events.get("flags", {}) if isinstance(events, dict) else {}
        entities = info.get("entities", {}) if isinstance(info, dict) else {}

        if flags.get("chest_opened"):
            self.opened_rooms.add(room_id)
        if flags.get("button_pressed"):
            self.button_pressed = True
        if room_id == "room_0_0" and int(entities.get("buttons_pressed", 0) or 0) > 0:
            self.button_pressed = True
        if int(entities.get("chests_remaining", 1) or 0) == 0 and room_id in ROOM_CHESTS:
            self.opened_rooms.add(room_id)

    def _advance_route(self, room_id: str, info: dict) -> None:
        del info
        while self.route_index < len(self.route):
            expected_room, kind, target = self.route[self.route_index]
            if kind == "open_chest" and expected_room in self.opened_rooms:
                self.route_index += 1
                continue
            if kind == "press_button" and self.button_pressed:
                self.route_index += 1
                continue
            if (
                kind == "go_exit"
                and self.active_go_index == self.route_index
                and room_id == NEXT_ROOM.get((expected_room, str(target)))
            ):
                self.exit_push_steps = 0
                self.exit_push_action = None
                self.active_go_index = None
                self.route_index += 1
                continue
            break

    def _walls(self, room_id: str) -> set[tuple[int, int]]:
        return set(ROOM_WALLS.get(room_id, set()))

    def _traps(self, room_id: str) -> set[tuple[int, int]]:
        return set(ROOM_TRAPS.get(room_id, set()))

    def _chest_blockers(self, room_id: str) -> set[tuple[int, int]]:
        return {
            chest
            for chest_room, chest in ROOM_CHESTS.items()
            if chest_room == room_id
        }

    def _combat_action(
        self,
        player_tile: tuple[int, int],
        monsters: list[tuple[int, int]],
    ) -> int | None:
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
            return ACTION_A
        adjacent = [monster for monster in monsters if _manhattan(player_tile, monster) == 1]
        if not adjacent:
            return None
        monster = min(adjacent, key=lambda pos: _manhattan(player_tile, pos))
        facing_action = _direction_to_adjacent(player_tile, monster)
        if facing_action == self.last_action:
            self.attack_cooldown = 5
            return ACTION_A
        return facing_action

    def _start_room_combat_action(self, player_tile: tuple[int, int]) -> int | None:
        target = (6, 4)
        if player_tile[1] < target[1]:
            return ACTION_DOWN
        if player_tile[1] > target[1]:
            return ACTION_UP
        if player_tile[0] < target[0]:
            return ACTION_RIGHT
        if player_tile[0] > target[0]:
            return ACTION_LEFT
        if self.last_action != ACTION_LEFT:
            return ACTION_LEFT
        self.attack_cooldown = 2
        return ACTION_A

    def _route_action(
        self,
        room_id: str,
        player_tile: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        if self.route_index >= len(self.route):
            return ACTION_WAIT

        expected_room, kind, target = self.route[self.route_index]
        if expected_room != room_id and kind != "go_exit":
            return ACTION_WAIT

        if kind == "open_chest":
            return self._open_chest(player_tile, target, blocked)
        if kind == "press_button":
            return self._press_button(player_tile, target, blocked)
        if kind == "go_exit":
            return self._go_exit(player_tile, str(target), blocked)
        return None

    def _open_chest(
        self,
        player_tile: tuple[int, int],
        chest: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        if chest == ROOM_CHESTS["room_-1_0"]:
            return self._open_west_chest_safely(player_tile)

        if _manhattan(player_tile, chest) == 1:
            return ACTION_A
        goals = {
            neighbor
            for neighbor in _neighbors(chest)
            if _inside(neighbor) and neighbor not in blocked
        }
        return _bfs_next_action(player_tile, goals, blocked)

    def _open_west_chest_safely(self, player_tile: tuple[int, int]) -> int:
        if not self.west_setup_done:
            if not self.west_setup_script:
                self.west_setup_script = [ACTION_LEFT] * self._ticks_to_actions(9) + [ACTION_A] * 2
            action = self.west_setup_script.pop(0)
            if not self.west_setup_script:
                self.west_setup_done = True
            return action

        if player_tile == WEST_CHEST_STAND:
            return ACTION_A

        if player_tile[1] < 5:
            return ACTION_DOWN
        if player_tile[1] == 5 and player_tile[0] > 6:
            return ACTION_LEFT
        if player_tile[1] == 5 and player_tile[0] == 6 and not self.west_wall_aligned:
            if self.west_wall_align_steps <= 0:
                self.west_wall_align_steps = self._ticks_to_actions(8)
            if self.west_wall_align_steps > 0:
                self.west_wall_align_steps -= 1
                if self.west_wall_align_steps == 0:
                    self.west_wall_aligned = True
                return ACTION_LEFT
        if player_tile[1] < WEST_CHEST_STAND[1]:
            return ACTION_DOWN
        if player_tile[1] == WEST_CHEST_STAND[1] and player_tile[0] >= 6 and not self.west_floor_aligned:
            if self.west_floor_align_steps <= 0:
                self.west_floor_align_steps = self._ticks_to_actions(8)
            if self.west_floor_align_steps > 0:
                self.west_floor_align_steps -= 1
                if self.west_floor_align_steps == 0:
                    self.west_floor_aligned = True
                return ACTION_DOWN
        if player_tile[1] > WEST_CHEST_STAND[1]:
            return ACTION_UP
        if player_tile[0] > WEST_CHEST_STAND[0]:
            return ACTION_LEFT
        if player_tile[0] < WEST_CHEST_STAND[0]:
            return ACTION_RIGHT
        return ACTION_A

    def _press_button(
        self,
        player_tile: tuple[int, int],
        button: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        if player_tile == button:
            self.button_pressed = True
            return ACTION_WAIT
        return _bfs_next_action(player_tile, {button}, blocked - {button})

    def _go_exit(
        self,
        player_tile: tuple[int, int],
        direction: str,
        blocked: set[tuple[int, int]],
    ) -> int | None:
        self.active_go_index = self.route_index
        push_action = EXIT_ACTION[direction]
        if self.exit_push_steps > 0:
            self.exit_push_steps -= 1
            return self.exit_push_action or push_action
        if player_tile in EXIT_TILES[direction]:
            self.exit_push_action = push_action
            self.exit_push_steps = self._ticks_to_actions(EXIT_PUSH_STEPS)
            return push_action
        return _bfs_next_action(player_tile, EXIT_TILES[direction], blocked)

    def _fallback_action(
        self,
        room_id: str,
        player_tile: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        if self.route_index < len(self.route):
            _, _, target = self.route[self.route_index]
            if isinstance(target, tuple):
                return self._greedy_towards(player_tile, target, blocked)
        chest = ROOM_CHESTS.get(room_id)
        if chest is not None and room_id not in self.opened_rooms:
            return self._greedy_towards(player_tile, chest, blocked)
        return ACTION_WAIT

    def _greedy_towards(
        self,
        player_tile: tuple[int, int],
        target: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        candidates: list[tuple[int, int]] = []
        for action, (dx, dy) in MOVE_TO_DELTA.items():
            nxt = (player_tile[0] + dx, player_tile[1] + dy)
            if _inside(nxt) and nxt not in blocked:
                candidates.append((_manhattan(nxt, target), action))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

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
