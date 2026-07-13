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
        - Normalize per-image (mean/std) to handle brightness/contrast shifts.
        - Clip to [0,1] after normalization to avoid overflow.

        Args:
            pixel_obs: RGB image, shape (128, 160, 3), dtype uint8

        Returns:
            tensor, shape (1, 3, 128, 160)
        """
        if pixel_obs.shape != (128, 160, 3):
            raise ValueError(
                f"expected pixel_obs shape (128, 160, 3), got {pixel_obs.shape}"
            )

        x = pixel_obs.astype(np.float32) / 255.0

        # ✅ Per-image normalization: robust to color/brightness shifts
        mean = x.mean(axis=(0, 1), keepdims=True)
        std = x.std(axis=(0, 1), keepdims=True)
        x = (x - mean) / (std + 1e-6)

        x = np.clip(x, -1.0, 1.0)  # prevent extreme outliers after norm

        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        x = x.to(self.device)
        return x

    @torch.no_grad()
    def predict_logits(self, pixel_obs: np.ndarray) -> torch.Tensor:
        x = self._preprocess(pixel_obs)
        logits = self.model(x)
        return logits

    @torch.no_grad()
    def predict_grid(self, pixel_obs: np.ndarray) -> np.ndarray:
        """
        Returns:
            grid: np.ndarray, shape (8, 10)

        注意：
            grid 访问方式是 grid[y, x]
            对外 tile 坐标统一返回 [x, y]
        """
        logits = self.predict_logits(pixel_obs)
        grid = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return grid

    @torch.no_grad()
    def predict_grid_with_confidence(self, pixel_obs: np.ndarray) -> dict:
        logits = self.predict_logits(pixel_obs)

        prob = torch.softmax(logits, dim=1)          # (1, C, 8, 10)
        confidence, pred = prob.max(dim=1)          # (1, 8, 10)

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
        从类别概率图中找一个最可能的位置。
        提高鲁棒性: 若置信度低，返回 [-1, -1]（不误导策略）
        """
        if class_id < 0 or class_id >= prob.shape[0]:
            return np.array([-1, -1], dtype=np.int32), 0.0

        class_prob = prob[class_id]  # shape (8, 10), index as [y, x]

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
        从 argmax grid 中找所有某个类别的位置。
        """
        positions_yx = np.argwhere(grid == class_id)

        if len(positions_yx) == 0:
            return np.empty((0, 2), dtype=np.int32)

        positions_xy = np.zeros((len(positions_yx), 2), dtype=np.int32)
        for idx, (y, x) in enumerate(positions_yx):
            positions_xy[idx] = np.array([x, y], dtype=np.int32)

        return positions_xy

    def _find_all_objects_from_grid_multi_id_xy(
        self,
        grid: np.ndarray,
        class_ids: list[int],
    ) -> np.ndarray:
        """
        从 grid 中找多个类别 id 的所有位置
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
        把变长坐标列表 padding 成固定长度。
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
        player_threshold: float = 0.7,
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
        Args:
            pixel_obs: RGB image, shape (128, 160, 3), dtype uint8

        Returns:
            所有坐标均为 [x, y]，和 nesylink 官方 tile 坐标一致。

        ✅ 鲁棒性改进：
          - player_threshold 默认 0.7，避免在 redraw/color 下误检
          - doors_all = exits_all（统一门语义）
          - 不再输出 locked/unlocked_doors（已删除）
        """
        result = self.predict_grid_with_confidence(pixel_obs)

        grid = result["grid"]
        confidence = result["confidence"]
        prob = result["prob"]

        # 玩家：用概率最大位置 + 高阈值防误检（color/redraw 下常见误检）
        player_tile, player_confidence = self._find_single_object_by_prob_xy(
            prob=prob,
            class_id=PLAYER_ID,
            threshold=player_threshold,
        )

        # 多实例对象：从 argmax grid 中提取
        walls_all = self._find_all_objects_from_grid_xy(grid, WALL_ID)
        monsters_all = self._find_all_objects_from_grid_xy(grid, MONSTER_ID)
        chests_all = self._find_all_objects_from_grid_xy(grid, CHEST_ID)
        exits_all = self._find_all_objects_from_grid_xy(grid, EXIT_ID)
        traps_all = self._find_all_objects_from_grid_xy(grid, TRAP_ID)
        buttons_all = self._find_all_objects_from_grid_xy(grid, BUTTON_ID)
        npcs_all = self._find_all_objects_from_grid_xy(grid, NPC_ID)
        gaps_all = self._find_all_objects_from_grid_xy(grid, GAP_ID)
        bridges_all = self._find_all_objects_from_grid_xy(grid, BRIDGE_ID)
        switches_all = self._find_all_objects_from_grid_xy(grid, SWITCH_ID)

        # ============================================================
        # ✅ 门 / 出入口：统一处理，不区分 locked/unlocked
        # ============================================================
        doors_all = exits_all  # 仅此一行，已符合你要求

        # 机关合并字段
        mechanisms_all = self._find_all_objects_from_grid_multi_id_xy(
            grid,
            [BUTTON_ID, SWITCH_ID],
        )

        # 桥相关合并字段
        bridge_tiles_all = self._find_all_objects_from_grid_multi_id_xy(
            grid,
            [GAP_ID, BRIDGE_ID],
        )

        # padding
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

        # 统计量
        monsters_remaining = int(len(monsters_all))
        chests_remaining = int(len(chests_all))
        doors_remaining = int(len(doors_all))
        traps_active = int(len(traps_all))

        # 桥房间检测
        is_bridge_room = bool(
            len(gaps_all) > 0
            or len(bridges_all) > 0
            or len(switches_all) > 0
        )

        return {
            # 原始预测
            "grid": grid,
            "confidence": confidence,

            # 玩家（加阈值鲁棒）
            "player_tile": player_tile,
            "player_confidence": player_confidence,

            # 怪物
            "monsters_tile": monsters_tile,
            "monsters_active_mask": monsters_active_mask,
            "monsters_all": monsters_all,
            "monsters_remaining": monsters_remaining,

            # 门：统一字段
            "doors_tile": doors_tile,
            "doors_active_mask": doors_active_mask,
            "doors_all": doors_all,
            "doors_remaining": doors_remaining,

            # 兼容 exit（保留，下游可删）
            "exits_tile": doors_tile,
            "exits_active_mask": doors_active_mask,
            "exits_all": doors_all,
            "exits_remaining": doors_remaining,

            # 墙
            "walls_tile": walls_tile,
            "walls_active_mask": walls_active_mask,
            "walls_all": walls_all,

            # 宝箱
            "chests_tile": chests_tile,
            "chests_active_mask": chests_active_mask,
            "chests_all": chests_all,
            "chests_remaining": chests_remaining,

            # 陷阱
            "traps_tile": traps_tile,
            "traps_active_mask": traps_active_mask,
            "traps_all": traps_all,
            "traps_active": traps_active,

            # 机关
            "buttons_tile": buttons_tile,
            "buttons_active_mask": buttons_active_mask,
            "buttons_all": buttons_all,
            "switches_tile": switches_tile,
            "switches_active_mask": switches_active_mask,
            "switches_all": switches_all,
            "mechanisms_all": mechanisms_all,

            # NPC
            "npcs_tile": npcs_tile,
            "npcs_active_mask": npcs_active_mask,
            "npcs_all": npcs_all,

            # 桥
            "gaps_tile": gaps_tile,
            "gaps_active_mask": gaps_active_mask,
            "gaps_all": gaps_all,
            "bridges_tile": bridges_tile,
            "bridges_active_mask": bridges_active_mask,
            "bridges_all": bridges_all,
            "bridge_tiles_all": bridge_tiles_all,
            "is_bridge_room": is_bridge_room,
        }