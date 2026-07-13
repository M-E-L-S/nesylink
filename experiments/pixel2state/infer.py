from pathlib import Path

import numpy as np
import torch

from experiments.pixel2state.model import make_model

# ============================================================
# Tile IDs from nesylink observation.py
# ============================================================

EMPTY_ID = 0
WALL_ID = 1
PLAYER_ID = 2
MONSTER_ID = 3
CHEST_ID = 4
EXIT_ID = 5
TRAP_ID = 6
BUTTON_ID = 7
NPC_ID = 8
GAP_ID = 9
BRIDGE_ID = 10
SWITCH_ID = 11

class PixelToStatePredictor:
    def __init__(self, model_path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.num_tile_classes = checkpoint.get("num_tile_classes", 12)

        if self.num_tile_classes <= SWITCH_ID:
            raise ValueError(
                f"num_tile_classes={self.num_tile_classes} is too small. "
                f"Need at least {SWITCH_ID + 1} classes for tile id {SWITCH_ID}."
            )

        self.model = make_model(num_tile_classes=self.num_tile_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, pixel_obs: np.ndarray) -> torch.Tensor:
        """
        Robust preprocessing for color/spatial/redraw variants.
        """
        if pixel_obs.shape != (128, 160, 3):
            raise ValueError(
                f"expected pixel_obs shape (128, 160, 3), got {pixel_obs.shape}"
            )

        x = pixel_obs.astype(np.float32) / 255.0

        mean = x.mean(axis=(0, 1), keepdims=True)
        std = x.std(axis=(0, 1), keepdims=True)
        x = (x - mean) / (std + 1e-6)
        x = np.clip(x, -1.0, 1.0)

        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return x.to(self.device)

    @torch.no_grad()
    def predict_logits(self, pixel_obs: np.ndarray) -> torch.Tensor:
        x = self._preprocess(pixel_obs)
        return self.model(x)

    @torch.no_grad()
    def predict_grid(self, pixel_obs: np.ndarray) -> np.ndarray:
        """
        Returns:
            grid: np.ndarray, shape (8, 10)

        grid access is grid[y, x].
        Public tile coordinates are [x, y].
        """
        logits = self.predict_logits(pixel_obs)
        grid = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return grid

    @torch.no_grad()
    def predict_grid_with_confidence(self, pixel_obs: np.ndarray) -> dict:
        logits = self.predict_logits(pixel_obs)

        prob = torch.softmax(logits, dim=1)
        confidence, pred = prob.max(dim=1)

        grid = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        confidence = confidence.squeeze(0).cpu().numpy().astype(np.float32)
        prob = prob.squeeze(0).cpu().numpy().astype(np.float32)

        return {
            "grid": grid,
            "confidence": confidence,
            "prob": prob,
        }

    def _find_single_object_by_prob_xy(
        self,
        prob: np.ndarray,
        class_id: int,
        threshold: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """
        Find the most likely tile for one object class.
        """
        if class_id < 0 or class_id >= prob.shape[0]:
            return np.array([-1, -1], dtype=np.int32), 0.0

        class_prob = prob[class_id]

        flat_idx = int(class_prob.argmax())
        y, x = np.unravel_index(flat_idx, class_prob.shape)
        conf = float(class_prob[y, x])

        if conf < threshold:
            return np.array([-1, -1], dtype=np.int32), conf

        return np.array([x, y], dtype=np.int32), conf

    def _find_all_objects_from_grid_xy(
        self,
        grid: np.ndarray,
        class_id: int,
    ) -> np.ndarray:
        """
        Find all positions of a class from argmax grid.
        """
        positions_yx = np.argwhere(grid == class_id)

        if len(positions_yx) == 0:
            return np.empty((0, 2), dtype=np.int32)

        positions_xy = np.zeros((len(positions_yx), 2), dtype=np.int32)
        for idx, (y, x) in enumerate(positions_yx):
            positions_xy[idx] = np.array([x, y], dtype=np.int32)

        return positions_xy

    def _find_objects_by_grid_or_prob_xy(
        self,
        grid: np.ndarray,
        prob: np.ndarray,
        class_id: int,
        max_count: int,
        prob_threshold: float = 0.12,
        grid_threshold: float | None = None,
    ) -> np.ndarray:
        """
        Robust object extraction:
        1. Use argmax grid positions only when their class probability is credible.
        2. Supplement from probability map only when confidence is high enough.
        """
        if class_id < 0 or class_id >= prob.shape[0]:
            return np.empty((0, 2), dtype=np.int32)

        if grid_threshold is None:
            grid_threshold = prob_threshold

        found = []
        class_prob = prob[class_id]

        grid_positions = self._find_all_objects_from_grid_xy(grid, class_id)
        for pos in grid_positions:
            x, y = int(pos[0]), int(pos[1])
            if x < 0 or y < 0:
                continue

            score = float(class_prob[y, x])
            if score < grid_threshold:
                continue

            if (x, y) not in found:
                found.append((x, y))

            if len(found) >= max_count:
                break

        flat_order = np.argsort(class_prob.reshape(-1))[::-1]

        for flat_idx in flat_order:
            y, x = np.unravel_index(int(flat_idx), class_prob.shape)
            score = float(class_prob[y, x])

            if score < prob_threshold:
                break

            if (int(x), int(y)) not in found:
                found.append((int(x), int(y)))

            if len(found) >= max_count:
                break

        if not found:
            return np.empty((0, 2), dtype=np.int32)

        return np.array(found[:max_count], dtype=np.int32)

    def _tiles_from_color_mask_xy(
        self,
        mask: np.ndarray,
        min_pixels: int,
        max_count: int,
    ) -> np.ndarray:
        """
        Convert a pixel-level color mask into tile coordinates.

        Board is 10 x 8 tiles, each tile is 16 x 16 pixels.
        Public coordinates are [x, y].
        """
        found = []

        for ty in range(8):
            y0 = ty * 16
            y1 = y0 + 16

            for tx in range(10):
                x0 = tx * 16
                x1 = x0 + 16

                tile_mask = mask[y0:y1, x0:x1]
                count = int(tile_mask.sum())

                if count >= min_pixels:
                    found.append((tx, ty, count))

        if not found:
            return np.empty((0, 2), dtype=np.int32)

        found.sort(key=lambda item: item[2], reverse=True)
        found = found[:max_count]

        return np.array([[x, y] for x, y, _ in found], dtype=np.int32)

    def _best_tile_from_color_mask_xy(
        self,
        mask: np.ndarray,
        min_pixels: int,
    ) -> tuple[np.ndarray, int]:
        """
        Return the tile with the strongest mask evidence.
        """
        best_xy = np.array([-1, -1], dtype=np.int32)
        best_count = 0

        for ty in range(8):
            y0 = ty * 16
            y1 = y0 + 16

            for tx in range(10):
                x0 = tx * 16
                x1 = x0 + 16

                count = int(mask[y0:y1, x0:x1].sum())
                if count > best_count:
                    best_count = count
                    best_xy = np.array([tx, ty], dtype=np.int32)

        if best_count < min_pixels:
            return np.array([-1, -1], dtype=np.int32), best_count

        return best_xy, best_count

    def _looks_like_redraw_obs(self, pixel_obs: np.ndarray) -> bool:
        """
        Conservatively detect redraw-style observations from raw pixels.

        This should trigger for geometric/symbolic redraw, but stay false for
        original/spatial/grayscale.
        """
        if pixel_obs.shape != (128, 160, 3):
            return False

        img = pixel_obs.astype(np.int16)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        white_mask = (r >= 220) & (g >= 220) & (b >= 220)
        black_mask = (r <= 25) & (g <= 25) & (b <= 25)

        cyan_mask = (
            (g >= 120)
            & (b >= 120)
            & (r <= 120)
            & (g >= r + 35)
            & (b >= r + 35)
        )

        red_mask = (
            (r >= 140)
            & (r >= g + 55)
            & (r >= b + 55)
            & (g <= 140)
            & (b <= 140)
        )

        yellow_mask = (
            (r >= 150)
            & (g >= 105)
            & (b <= 125)
            & (r >= b + 50)
            & (g >= b + 35)
        )

        white_tiles = 0
        black_tiles = 0
        cyan_tiles = 0
        red_yellow_tiles = 0

        for ty in range(8):
            y0 = ty * 16
            y1 = y0 + 16

            for tx in range(10):
                x0 = tx * 16
                x1 = x0 + 16

                if int(white_mask[y0:y1, x0:x1].sum()) >= 110:
                    white_tiles += 1
                if int(black_mask[y0:y1, x0:x1].sum()) >= 110:
                    black_tiles += 1
                if int(cyan_mask[y0:y1, x0:x1].sum()) >= 14:
                    cyan_tiles += 1
                if (
                    int(red_mask[y0:y1, x0:x1].sum()) >= 16
                    or int(yellow_mask[y0:y1, x0:x1].sum()) >= 12
                ):
                    red_yellow_tiles += 1

        geometric_like = white_tiles >= 6 and cyan_tiles >= 1
        symbolic_like = black_tiles >= 8 and (cyan_tiles >= 1 or red_yellow_tiles >= 1)

        return bool(geometric_like or symbolic_like)

    def _detect_redraw_geometry_from_pixels_xy(
        self,
        pixel_obs: np.ndarray,
        max_monsters: int,
        max_chests: int,
        max_walls: int,
    ) -> dict:
        """
        Redraw-only pixel parser.

        Geometric redraw:
        - wall: white square
        - chest: yellow diamond
        - agent: cyan circle + direction triangle
        - monster: red hexagon

        Symbolic redraw:
        - wall: black square
        - chest: gold/yellow symbol
        - monster: red symbol block
        """
        empty = {
            "player_tile": np.array([-1, -1], dtype=np.int32),
            "walls_all": np.empty((0, 2), dtype=np.int32),
            "monsters_all": np.empty((0, 2), dtype=np.int32),
            "chests_all": np.empty((0, 2), dtype=np.int32),
        }

        if pixel_obs.shape != (128, 160, 3):
            return empty

        img = pixel_obs.astype(np.int16)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        white_mask = (r >= 220) & (g >= 220) & (b >= 220)
        black_mask = (r <= 25) & (g <= 25) & (b <= 25)

        red_mask = (
            (r >= 120)
            & (r >= g + 45)
            & (r >= b + 45)
            & (g <= 155)
            & (b <= 155)
        )

        yellow_mask = (
            (r >= 135)
            & (g >= 90)
            & (b <= 135)
            & (r >= b + 40)
            & (g >= b + 30)
        )

        cyan_mask = (
            (g >= 120)
            & (b >= 120)
            & (r <= 120)
            & (g >= r + 35)
            & (b >= r + 35)
        )

        white_walls = self._tiles_from_color_mask_xy(
            mask=white_mask,
            min_pixels=110,
            max_count=max_walls,
        )
        black_walls = self._tiles_from_color_mask_xy(
            mask=black_mask,
            min_pixels=110,
            max_count=max_walls,
        )

        if len(white_walls) >= len(black_walls):
            walls = white_walls
        else:
            walls = black_walls

        monsters = self._tiles_from_color_mask_xy(
            mask=red_mask,
            min_pixels=16,
            max_count=max_monsters,
        )
        chests = self._tiles_from_color_mask_xy(
            mask=yellow_mask,
            min_pixels=12,
            max_count=max_chests,
        )
        player_tile, _ = self._best_tile_from_color_mask_xy(
            mask=cyan_mask,
            min_pixels=14,
        )

        return {
            "player_tile": player_tile,
            "walls_all": walls,
            "monsters_all": monsters,
            "chests_all": chests,
        }

    def _merge_positions_xy(
        self,
        primary: np.ndarray,
        secondary: np.ndarray,
        max_count: int,
    ) -> np.ndarray:
        """
        Merge coordinate lists while preserving order and removing duplicates.
        """
        merged = []

        for arr in (primary, secondary):
            if arr is None or len(arr) == 0:
                continue

            for pos in arr:
                x, y = int(pos[0]), int(pos[1])
                if x < 0 or y < 0:
                    continue

                if (x, y) not in merged:
                    merged.append((x, y))

                if len(merged) >= max_count:
                    break

            if len(merged) >= max_count:
                break

        if not merged:
            return np.empty((0, 2), dtype=np.int32)

        return np.array(merged, dtype=np.int32)

    def _find_all_objects_from_grid_multi_id_xy(
        self,
        grid: np.ndarray,
        class_ids: list[int],
    ) -> np.ndarray:
        """
        Find all positions matching any class id.
        """
        all_positions = []

        for class_id in class_ids:
            positions = self._find_all_objects_from_grid_xy(grid, class_id)
            if len(positions) > 0:
                all_positions.append(positions)

        if not all_positions:
            return np.empty((0, 2), dtype=np.int32)

        return np.concatenate(all_positions, axis=0).astype(np.int32)

    def _pad_positions(
        self,
        positions_xy: np.ndarray,
        max_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pad variable-length coordinate list to fixed length.
        """
        padded = np.full((max_count, 2), -1, dtype=np.int32)
        active_mask = np.zeros((max_count,), dtype=bool)

        n = min(len(positions_xy), max_count)
        if n > 0:
            padded[:n] = positions_xy[:n]
            active_mask[:n] = True

        return padded, active_mask

    def predict_state(
        self,
        pixel_obs: np.ndarray,
        player_threshold: float = 0.25,
        max_monsters: int = 8,
        max_doors: int = 8,
        max_walls: int = 80,
        max_chests: int = 8,
        max_buttons: int = 8,
        max_switches: int = 8,
        max_npcs: int = 8,
        max_traps: int = 32,
        max_gaps: int = 32,
        max_bridges: int = 16,
    ) -> dict:
        """
        Returns:
            All coordinates are [x, y], matching official tile coordinates.
        """
        result = self.predict_grid_with_confidence(pixel_obs)

        grid = result["grid"]
        confidence = result["confidence"]
        prob = result["prob"]

        player_tile, player_confidence = self._find_single_object_by_prob_xy(
            prob=prob,
            class_id=PLAYER_ID,
            threshold=player_threshold,
        )

        # Large/background-like objects stay grid-based by default.
        walls_all = self._find_all_objects_from_grid_xy(grid, WALL_ID)
        traps_all = self._find_all_objects_from_grid_xy(grid, TRAP_ID)
        buttons_all = self._find_all_objects_from_grid_xy(grid, BUTTON_ID)
        npcs_all = self._find_all_objects_from_grid_xy(grid, NPC_ID)
        gaps_all = self._find_all_objects_from_grid_xy(grid, GAP_ID)
        bridges_all = self._find_all_objects_from_grid_xy(grid, BRIDGE_ID)
        switches_all = self._find_all_objects_from_grid_xy(grid, SWITCH_ID)

        # Critical small objects use grid + probability fallback.
        monsters_model = self._find_objects_by_grid_or_prob_xy(
            grid=grid,
            prob=prob,
            class_id=MONSTER_ID,
            max_count=max_monsters,
            prob_threshold=0.18,
            grid_threshold=0.18,
        )
        chests_model = self._find_objects_by_grid_or_prob_xy(
            grid=grid,
            prob=prob,
            class_id=CHEST_ID,
            max_count=max_chests,
            prob_threshold=0.18,
            grid_threshold=0.18,
        )
        exits_all = self._find_objects_by_grid_or_prob_xy(
            grid=grid,
            prob=prob,
            class_id=EXIT_ID,
            max_count=max_doors,
            prob_threshold=0.12,
            grid_threshold=0.12,
        )

        if self._looks_like_redraw_obs(pixel_obs):
            redraw_detected = self._detect_redraw_geometry_from_pixels_xy(
                pixel_obs=pixel_obs,
                max_monsters=max_monsters,
                max_chests=max_chests,
                max_walls=max_walls,
            )

            if int(redraw_detected["player_tile"][0]) >= 0:
                player_tile = redraw_detected["player_tile"]
                player_confidence = 1.0

            if len(redraw_detected["walls_all"]) >= 4:
                walls_all = redraw_detected["walls_all"]

            # In redraw mode, pixel geometry is more reliable than sprite-trained
            # CNN output for monsters/chests, so use it first.
            monsters_all = self._merge_positions_xy(
                primary=redraw_detected["monsters_all"],
                secondary=monsters_model,
                max_count=max_monsters,
            )
            chests_all = self._merge_positions_xy(
                primary=redraw_detected["chests_all"],
                secondary=chests_model,
                max_count=max_chests,
            )
        else:
            monsters_all = monsters_model
            chests_all = chests_model

        doors_all = exits_all

        mechanisms_all = self._find_all_objects_from_grid_multi_id_xy(
            grid,
            [BUTTON_ID, SWITCH_ID],
        )

        bridge_tiles_all = self._find_all_objects_from_grid_multi_id_xy(
            grid,
            [GAP_ID, BRIDGE_ID],
        )

        monsters_tile, monsters_active_mask = self._pad_positions(
            monsters_all,
            max_monsters,
        )
        doors_tile, doors_active_mask = self._pad_positions(
            doors_all,
            max_doors,
        )
        walls_tile, walls_active_mask = self._pad_positions(
            walls_all,
            max_walls,
        )
        chests_tile, chests_active_mask = self._pad_positions(
            chests_all,
            max_chests,
        )
        buttons_tile, buttons_active_mask = self._pad_positions(
            buttons_all,
            max_buttons,
        )
        switches_tile, switches_active_mask = self._pad_positions(
            switches_all,
            max_switches,
        )
        npcs_tile, npcs_active_mask = self._pad_positions(
            npcs_all,
            max_npcs,
        )
        traps_tile, traps_active_mask = self._pad_positions(
            traps_all,
            max_traps,
        )
        gaps_tile, gaps_active_mask = self._pad_positions(
            gaps_all,
            max_gaps,
        )
        bridges_tile, bridges_active_mask = self._pad_positions(
            bridges_all,
            max_bridges,
        )

        monsters_remaining = int(len(monsters_all))
        chests_remaining = int(len(chests_all))
        doors_remaining = int(len(doors_all))
        traps_active = int(len(traps_all))

        is_bridge_room = bool(
            len(gaps_all) > 0
            or len(bridges_all) > 0
            or len(switches_all) > 0
        )

        return {
            "grid": grid,
            "confidence": confidence,

            "player_tile": player_tile,
            "player_confidence": player_confidence,

            "monsters_tile": monsters_tile,
            "monsters_active_mask": monsters_active_mask,
            "monsters_all": monsters_all,
            "monsters_remaining": monsters_remaining,

            "doors_tile": doors_tile,
            "doors_active_mask": doors_active_mask,
            "doors_all": doors_all,
            "doors_remaining": doors_remaining,

            "exits_tile": doors_tile,
            "exits_active_mask": doors_active_mask,
            "exits_all": doors_all,
            "exits_remaining": doors_remaining,

            "walls_tile": walls_tile,
            "walls_active_mask": walls_active_mask,
            "walls_all": walls_all,

            "chests_tile": chests_tile,
            "chests_active_mask": chests_active_mask,
            "chests_all": chests_all,
            "chests_remaining": chests_remaining,

            "traps_tile": traps_tile,
            "traps_active_mask": traps_active_mask,
            "traps_all": traps_all,
            "traps_active": traps_active,

            "buttons_tile": buttons_tile,
            "buttons_active_mask": buttons_active_mask,
            "buttons_all": buttons_all,
            "switches_tile": switches_tile,
            "switches_active_mask": switches_active_mask,
            "switches_all": switches_all,
            "mechanisms_all": mechanisms_all,

            "npcs_tile": npcs_tile,
            "npcs_active_mask": npcs_active_mask,
            "npcs_all": npcs_all,

            "gaps_tile": gaps_tile,
            "gaps_active_mask": gaps_active_mask,
            "gaps_all": gaps_all,
            "bridges_tile": bridges_tile,
            "bridges_active_mask": bridges_active_mask,
            "bridges_all": bridges_all,
            "bridge_tiles_all": bridge_tiles_all,
            "is_bridge_room": is_bridge_room,
        }