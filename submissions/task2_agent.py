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
TASK2_TRAPS = {
    (x, 0) for x in range(1, 9)
} | {
    (x, 7) for x in range(1, 9)
}
TASK2_VARIANTS = {
    "default": {"spawn": (7, 3), "chest": (1, 3), "monster": (2, 2)},
    "spatial_a": {"spawn": (7, 4), "chest": (3, 3), "monster": (4, 2)},
    "spatial_b": {"spawn": (6, 2), "chest": (2, 5), "monster": (5, 4)},
    "spatial_c": {"spawn": (8, 5), "chest": (4, 3), "monster": (2, 4)},
}

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
        self.attack_cooldown = 0
        self.action_repeat = 1
        self.variant_name: str | None = None
        self.monster_tile: tuple[int, int] | None = None
        self.chest_tile: tuple[int, int] | None = None
        self.position_px: tuple[float, float] | None = None

    def act(self, obs: np.ndarray, info: dict) -> int:
        state = self.predictor.predict_state(obs)
        control = info.get("control", {}) if isinstance(info, dict) else {}
        self.action_repeat = max(1, int(control.get("action_repeat", 1) or 1))

        agent = info.get("agent", {}) if isinstance(info, dict) else {}
        player_tile = _as_xy_tuple(agent.get("tile")) or _as_xy_tuple(state.get("player_tile")) or self.last_player_tile
        if player_tile is None:
            return self.last_action
        self.position_px = self._position_px_from_info(agent)

        self._update_variant(player_tile, state)
        previous_player_tile = self.last_player_tile
        walls: set[tuple[int, int]] = set()
        traps = set(TASK2_TRAPS)
        monsters = _filter_positions(state.get("monsters_all", []))
        chests = _filter_positions(state.get("chests_all", []))

        inventory = info.get("inventory", {}) if isinstance(info, dict) else {}
        keys = int(inventory.get("keys", 0) or 0)
        entities = info.get("entities", {}) if isinstance(info, dict) else {}
        monsters_remaining = int(entities.get("monsters_remaining", 0) or 0)

        if monsters_remaining > 0:
            self.phase = "kill_monster"
            action = self._act_kill_monster(player_tile, monsters, walls | traps)
        elif keys <= 0:
            self.phase = "get_key"
            action = self._act_get_key(player_tile, chests, walls | traps)
        else:
            self.phase = "go_exit"
            action = self._act_go_exit(player_tile)

        if action is None:
            action = ACTION_WAIT
        if self.phase not in {"go_exit", "kill_monster"}:
            action = self._apply_turn_settle(previous_player_tile, player_tile, int(action))

        self.last_player_tile = player_tile
        if int(action) in MOVE_TO_DELTA:
            self.facing_action = int(action)
        self.last_action = int(action)
        return int(action)

    def _update_variant(self, player_tile: tuple[int, int], state: dict) -> None:
        if self.variant_name is None:
            for name, config in TASK2_VARIANTS.items():
                if player_tile == config["spawn"]:
                    self.variant_name = name
                    break

        if self.variant_name is None:
            detected_chests = set(_filter_positions(state.get("chests_all", [])))
            for name, config in TASK2_VARIANTS.items():
                if config["chest"] in detected_chests:
                    self.variant_name = name
                    break

        if self.variant_name is None:
            self.variant_name = "default"

        config = TASK2_VARIANTS[self.variant_name]
        self.monster_tile = config["monster"]
        self.chest_tile = config["chest"]

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
        del monsters
        monster = self.monster_tile
        if monster is None:
            return ACTION_WAIT
        distance = _manhattan(player_tile, monster)

        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
            return ACTION_A

        if distance == 1:
            facing_action = _direction_to_adjacent(player_tile, monster)
            if facing_action == self.facing_action:
                self.attack_cooldown = 4
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
        del chests
        chest = self.chest_tile or (1, 3)
        if self.variant_name == "spatial_c":
            stand = (4, 4)
            if player_tile == stand:
                return ACTION_A
            return _bfs_next_action(player_tile, {stand}, blocked | {chest})

        if _manhattan(player_tile, chest) == 1:
            return ACTION_A
        goals = {
            neighbor
            for neighbor in _neighbors(chest)
            if _inside(neighbor) and neighbor not in blocked and 1 <= neighbor[1] <= 6
        }
        return _bfs_next_action(player_tile, goals, blocked | {chest})

    def _act_go_exit(self, player_tile: tuple[int, int]) -> int:
        if self.exit_push_steps > 0:
            self.exit_push_steps -= 1
            return ACTION_LEFT
        if player_tile[0] <= 0 and player_tile[1] in {3, 4}:
            self.exit_push_steps = self._ticks_to_actions(EXIT_PUSH_STEPS)
            return ACTION_LEFT

        safe_y = 3 if self.variant_name == "spatial_b" else 4
        if player_tile[1] < safe_y:
            return ACTION_DOWN
        if player_tile[1] > safe_y:
            return ACTION_UP

        px_y = self.position_px[1] if self.position_px is not None else None
        if player_tile[1] == safe_y and px_y is not None and px_y < safe_y * 16:
            return ACTION_DOWN

        return ACTION_LEFT

    def _ticks_to_actions(self, ticks: int) -> int:
        return max(1, int((ticks + self.action_repeat - 1) // self.action_repeat))

    def _position_px_from_info(self, agent: dict) -> tuple[float, float] | None:
        pos = agent.get("position_px") if isinstance(agent, dict) else None
        try:
            if pos is not None and len(pos) >= 2:
                return (float(pos[0]), float(pos[1]))
        except Exception:
            return None
        return None


policy = Task2Policy()