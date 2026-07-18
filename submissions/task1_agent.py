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
MOVE_ACTIONS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}
DELTA_TO_ACTION = {delta: action for action, delta in MOVE_ACTIONS.items()}


def _as_tile(value) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        x = int(value[0])
        y = int(value[1])
    except Exception:
        return None
    if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
        return (x, y)
    return None


def _as_tiles(values) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    try:
        iterable = list(values)
    except Exception:
        return result
    for value in iterable:
        tile = _as_tile(value)
        if tile is not None and tile not in result:
            result.append(tile)
    return result


def _inside(tile: tuple[int, int]) -> bool:
    x, y = tile
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT


def _neighbors(tile: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = tile
    return [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _direction_action(src: tuple[int, int], dst: tuple[int, int]) -> int | None:
    return DELTA_TO_ACTION.get((dst[0] - src[0], dst[1] - src[1]))


def _bfs_next_action(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> int | None:
    if start in goals:
        return ACTION_WAIT

    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        current = queue.popleft()
        for nxt in _neighbors(current):
            if not _inside(nxt) or nxt in blocked or nxt in parent:
                continue
            parent[nxt] = current
            if nxt in goals:
                step = nxt
                while parent[step] is not None and parent[step] != start:
                    step = parent[step]
                return _direction_action(start, step)
            queue.append(nxt)
    return None


class Task1Policy:
    def __init__(self, model_path: str | None = None):
        resolved_model_path = model_path or str(Path("models/pixel2state/best.pt"))
        self.predictor = PixelToStatePredictor(resolved_model_path)
        self.reset()

    def reset(self, seed: int | None = None, task_id: str | None = None):
        del seed, task_id
        self.last_player_tile: tuple[int, int] | None = None
        self.last_move_action: int | None = None
        self.move_script: list[int] = []
        self.turn_then_open = False
        self.committed_start_tile: tuple[int, int] | None = None
        self.committed_action: int | None = None
        self.dynamic_blocked: set[tuple[int, int]] = set()
        self.stable_exits: list[tuple[int, int]] = []
        self.stable_exit_side: str | None = None
        self.exit_candidate_side: str | None = None
        self.exit_candidate_count = 0

    def act(self, obs: np.ndarray, info: dict) -> int:
        state = self._parse_state(obs)
        keys = int(info.get("inventory", {}).get("keys", 0) or 0)

        player_tile = self._resolve_player_tile(state["player_tile"])
        if player_tile is None:
            return self.last_move_action if self.last_move_action is not None else ACTION_WAIT

        self._finalize_committed_move(player_tile)

        chests = [tile for tile in state["chests"] if tile != player_tile]
        exit_candidates = [tile for tile in state["exits"] if tile != player_tile]
        exits = self._stabilize_exits(exit_candidates)
        walls = set(state["walls"]) | self.dynamic_blocked
        if keys <= 0:
            walls |= set(chests)

        if self.move_script:
            action = self.move_script.pop(0)
            self.last_move_action = action
            self.last_player_tile = player_tile
            return action

        if keys > 0:
            action = self._act_go_exit(player_tile, exits, walls)
        else:
            action = self._act_get_key(player_tile, chests, walls)

        if action in MOVE_ACTIONS:
            self.last_move_action = int(action)
            if not self.move_script and self._is_free_move(player_tile, int(action), walls):
                self.committed_start_tile = player_tile
                self.committed_action = int(action)
                self.move_script = [int(action)] * 15
            elif not self.move_script:
                self.committed_start_tile = None
                self.committed_action = None
        else:
            self.committed_start_tile = None
            self.committed_action = None
            self.last_move_action = None

        self.last_player_tile = player_tile
        return int(action)

    def _parse_state(self, obs: np.ndarray) -> dict[str, list[tuple[int, int]] | tuple[int, int] | None]:
        predicted = self.predictor.predict_state(obs)
        if not self._prediction_is_plausible(predicted):
            reinverted = self.predictor.predict_state(255 - obs)
            if self._prediction_is_plausible(reinverted):
                predicted = reinverted

        player_tile = _as_tile(predicted.get("player_tile"))
        walls = _as_tiles(predicted.get("walls_all", []))
        chests = _as_tiles(predicted.get("chests_all", []))
        exits = self._normalize_exit_tiles(_as_tiles(predicted.get("exits_all", [])))

        if self._should_use_redraw_detector(obs):
            redraw = self._detect_redraw_state(obs)
            if redraw["player_tile"] is not None:
                player_tile = redraw["player_tile"]
            if len(redraw["walls"]) >= 4:
                walls = redraw["walls"]
            if redraw["chests"] or not chests:
                chests = redraw["chests"]
            if redraw["exits"] or not exits:
                exits = redraw["exits"]

        chests = self._filter_chest_candidates(chests, exits)
        return {
            "player_tile": player_tile,
            "walls": walls,
            "chests": chests,
            "exits": exits,
        }

    def _should_use_redraw_detector(self, obs: np.ndarray) -> bool:
        pixels = obs.reshape(-1, obs.shape[-1]).astype(np.float32)
        unique_count = len(np.unique(pixels.astype(np.uint8), axis=0))
        channel_std = pixels.std(axis=0)
        return unique_count <= 8 and float(channel_std.max() - channel_std.min()) <= 15.0

    def _prediction_is_plausible(self, predicted: dict) -> bool:
        player_tile = _as_tile(predicted.get("player_tile"))
        walls = _as_tiles(predicted.get("walls_all", []))
        exits = self._normalize_exit_tiles(_as_tiles(predicted.get("exits_all", [])))
        if player_tile is None:
            return False
        if len(walls) < 8:
            return False
        if not exits:
            return False
        return True

    def _resolve_player_tile(self, predicted_player_tile: tuple[int, int] | None) -> tuple[int, int] | None:
        expected_tile = None
        if self.committed_start_tile is not None and self.committed_action in MOVE_ACTIONS and not self.move_script:
            expected_tile = self._expected_committed_tile()

        if predicted_player_tile is None:
            if expected_tile is not None:
                return expected_tile
            return self.last_player_tile
        if self.last_player_tile is None:
            return predicted_player_tile
        if expected_tile is not None:
            if predicted_player_tile == self.committed_start_tile:
                return predicted_player_tile
            return expected_tile
        if _manhattan(predicted_player_tile, self.last_player_tile) <= 1:
            return predicted_player_tile
        return self.last_player_tile

    def _is_free_move(
        self,
        player_tile: tuple[int, int],
        action: int,
        walls: set[tuple[int, int]],
    ) -> bool:
        dx, dy = MOVE_ACTIONS[action]
        target = (player_tile[0] + dx, player_tile[1] + dy)
        return _inside(target) and target not in walls

    def _expected_committed_tile(self) -> tuple[int, int] | None:
        if self.committed_start_tile is None or self.committed_action not in MOVE_ACTIONS:
            return None
        dx, dy = MOVE_ACTIONS[self.committed_action]
        tile = (self.committed_start_tile[0] + dx, self.committed_start_tile[1] + dy)
        if _inside(tile):
            return tile
        return None

    def _normalize_exit_tiles(self, exits: list[tuple[int, int]]) -> list[tuple[int, int]]:
        border_tiles = [tile for tile in exits if self._is_border_tile(tile)]
        if not border_tiles:
            return []

        sides: dict[str, list[tuple[int, int]]] = {
            "top": [],
            "bottom": [],
            "left": [],
            "right": [],
        }
        for tile in border_tiles:
            x, y = tile
            if y == 0:
                sides["top"].append(tile)
            elif y == GRID_HEIGHT - 1:
                sides["bottom"].append(tile)
            elif x == 0:
                sides["left"].append(tile)
            elif x == GRID_WIDTH - 1:
                sides["right"].append(tile)

        best: list[tuple[int, int]] = []
        for tiles in sides.values():
            if len(tiles) > len(best):
                best = tiles
        if not best:
            best = border_tiles
        return sorted(set(best))[:2]

    def _stabilize_exits(self, exits: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not exits:
            return list(self.stable_exits)

        side = self._exit_side(exits)
        if side is None:
            return list(self.stable_exits)

        if self.stable_exit_side is None:
            self.stable_exits = list(exits)
            self.stable_exit_side = side
            self.exit_candidate_side = None
            self.exit_candidate_count = 0
            return list(self.stable_exits)

        if side == self.stable_exit_side:
            if self._exit_sets_compatible(exits, self.stable_exits):
                merged = sorted(set(self.stable_exits) | set(exits))
                self.stable_exits = self._compact_exit_cluster(merged)
            self.exit_candidate_side = None
            self.exit_candidate_count = 0
            return list(self.stable_exits)

        if side == self.exit_candidate_side:
            self.exit_candidate_count += 1
        else:
            self.exit_candidate_side = side
            self.exit_candidate_count = 1

        if self.exit_candidate_count >= 4:
            self.stable_exits = list(exits)
            self.stable_exit_side = side
            self.exit_candidate_side = None
            self.exit_candidate_count = 0
        return list(self.stable_exits)

    def _exit_sets_compatible(
        self,
        exits: list[tuple[int, int]],
        stable_exits: list[tuple[int, int]],
    ) -> bool:
        return any(
            _manhattan(exit_tile, stable_tile) <= 1
            for exit_tile in exits
            for stable_tile in stable_exits
        )

    def _compact_exit_cluster(self, exits: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(exits) <= 2:
            return exits
        best_pair = min(
            ((a, b) for index, a in enumerate(exits) for b in exits[index + 1:]),
            key=lambda pair: _manhattan(pair[0], pair[1]),
        )
        return sorted(best_pair)

    def _exit_side(self, exits: list[tuple[int, int]]) -> str | None:
        side_counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        for x, y in exits:
            if y == 0:
                side_counts["top"] += 1
            elif y == GRID_HEIGHT - 1:
                side_counts["bottom"] += 1
            elif x == 0:
                side_counts["left"] += 1
            elif x == GRID_WIDTH - 1:
                side_counts["right"] += 1
        side, count = max(side_counts.items(), key=lambda item: item[1])
        if count <= 0:
            return None
        return side

    def _filter_chest_candidates(
        self,
        chests: list[tuple[int, int]],
        exits: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for chest in chests:
            if exits and any(_manhattan(chest, exit_tile) <= 1 for exit_tile in exits):
                continue
            result.append(chest)
        return result

    def _detect_redraw_state(self, obs: np.ndarray) -> dict[str, list[tuple[int, int]] | tuple[int, int] | None]:
        img = obs.astype(np.int16)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        white_mask = (r >= 220) & (g >= 220) & (b >= 220)
        black_mask = (r <= 25) & (g <= 25) & (b <= 25)
        cyan_mask = (g >= 120) & (b >= 120) & (r <= 120) & (g > r + 35) & (b > r + 35)
        yellow_mask = (r >= 135) & (g >= 90) & (b < 135) & (r > b + 40) & (g > b + 30)
        blue_mask = (b >= 70) & (b > r + 30) & (b > g + 10) & (r <= 110)

        white_walls = self._tiles_from_mask(white_mask, min_pixels=110)
        black_walls = self._tiles_from_mask(black_mask, min_pixels=110)
        walls = white_walls if len(white_walls) >= len(black_walls) else black_walls

        player_tile = self._best_tile_from_mask(cyan_mask, min_pixels=14)
        if player_tile is None:
            player_tile = self._best_tile_from_mask(white_mask, min_pixels=40)

        chests = self._tiles_from_mask(yellow_mask, min_pixels=12)
        exits = self._normalize_exit_tiles(self._tiles_from_mask(blue_mask, min_pixels=14))
        chests = self._filter_chest_candidates(chests, exits)
        return {
            "player_tile": player_tile,
            "walls": walls,
            "chests": chests,
            "exits": exits,
        }

    def _tiles_from_mask(self, mask: np.ndarray, min_pixels: int) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for ty in range(GRID_HEIGHT):
            for tx in range(GRID_WIDTH):
                count = int(mask[ty * 16:ty * 16 + 16, tx * 16:tx * 16 + 16].sum())
                if count >= min_pixels:
                    result.append((tx, ty))
        return result

    def _best_tile_from_mask(self, mask: np.ndarray, min_pixels: int) -> tuple[int, int] | None:
        best_tile: tuple[int, int] | None = None
        best_count = 0
        for ty in range(GRID_HEIGHT):
            for tx in range(GRID_WIDTH):
                count = int(mask[ty * 16:ty * 16 + 16, tx * 16:tx * 16 + 16].sum())
                if count > best_count:
                    best_count = count
                    best_tile = (tx, ty)
        if best_count < min_pixels:
            return None
        return best_tile

    def _is_border_tile(self, tile: tuple[int, int]) -> bool:
        x, y = tile
        return x == 0 or x == GRID_WIDTH - 1 or y == 0 or y == GRID_HEIGHT - 1

    def _finalize_committed_move(self, player_tile: tuple[int, int]) -> None:
        if self.move_script:
            return
        if self.committed_start_tile is None or self.committed_action not in MOVE_ACTIONS:
            return
        if player_tile == self.committed_start_tile:
            dx, dy = MOVE_ACTIONS[self.committed_action]
            blocked_tile = (player_tile[0] + dx, player_tile[1] + dy)
            if _inside(blocked_tile):
                self.dynamic_blocked.add(blocked_tile)
        else:
            self.dynamic_blocked.clear()
        self.committed_start_tile = None
        self.committed_action = None

    def _act_get_key(
        self,
        player_tile: tuple[int, int],
        chests: list[tuple[int, int]],
        walls: set[tuple[int, int]],
    ) -> int:
        if not chests:
            return ACTION_WAIT
        chest = min(chests, key=lambda tile: _manhattan(player_tile, tile))
        if _manhattan(player_tile, chest) == 1:
            if not self.turn_then_open:
                self.turn_then_open = True
                turn_action = _direction_action(player_tile, chest)
                if turn_action is not None:
                    return turn_action
            self.turn_then_open = False
            return ACTION_A

        self.turn_then_open = False
        stand_goals = {
            neighbor
            for neighbor in _neighbors(chest)
            if _inside(neighbor) and neighbor not in walls
        }
        action = _bfs_next_action(player_tile, stand_goals, walls)
        if action is not None:
            return action
        return self._greedy_move(player_tile, stand_goals, walls)

    def _act_go_exit(
        self,
        player_tile: tuple[int, int],
        exits: list[tuple[int, int]],
        walls: set[tuple[int, int]],
    ) -> int:
        if not exits:
            return ACTION_WAIT
        exit_goals = set(exits)
        if player_tile in exit_goals:
            push_action = self._exit_push_direction(player_tile)
            self.move_script = [push_action] * 19
            return push_action

        action = _bfs_next_action(player_tile, exit_goals, walls)
        if action is not None:
            return action
        return self._greedy_move(player_tile, exit_goals, walls)

    def _exit_push_direction(self, exit_tile: tuple[int, int]) -> int:
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

    def _greedy_move(
        self,
        player_tile: tuple[int, int],
        goals: set[tuple[int, int]],
        walls: set[tuple[int, int]],
    ) -> int:
        if not goals:
            return ACTION_WAIT
        target = min(goals, key=lambda tile: _manhattan(player_tile, tile))
        candidates: list[tuple[int, int]] = []
        for action, (dx, dy) in MOVE_ACTIONS.items():
            nxt = (player_tile[0] + dx, player_tile[1] + dy)
            if _inside(nxt) and nxt not in walls:
                candidates.append((_manhattan(nxt, target), action))
        if not candidates:
            return ACTION_WAIT
        candidates.sort()
        return candidates[0][1]


policy = Task1Policy()
