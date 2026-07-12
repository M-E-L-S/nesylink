from pathlib import Path

import numpy as np
import torch

from experiments.pixel2state.model import make_model

# 这里一定要确认和环境 full grid 里的真实类别编号一致
PLAYER_ID = 2
MONSTER_ID = 3

class PixelToStatePredictor:
    def __init__(self, model_path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.num_tile_classes = checkpoint.get("num_tile_classes", 12)

        self.model = make_model(num_tile_classes=self.num_tile_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, pixel_obs: np.ndarray) -> torch.Tensor:
        """
        Args:
            pixel_obs: RGB image, shape (128, 160, 3), dtype uint8

        Returns:
            x: tensor, shape (1, 3, 128, 160)
        """
        if pixel_obs.shape != (128, 160, 3):
            raise ValueError(
                f"expected pixel_obs shape (128, 160, 3), got {pixel_obs.shape}"
            )

        x = pixel_obs.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        x = x.to(self.device)
        return x

    @torch.no_grad()
    def predict_logits(self, pixel_obs: np.ndarray) -> torch.Tensor:
        """
        Returns:
            logits: tensor, shape (1, C, 8, 10)
        """
        x = self._preprocess(pixel_obs)
        logits = self.model(x)
        return logits

    @torch.no_grad()
    def predict_grid(self, pixel_obs: np.ndarray) -> np.ndarray:
        """
        Returns:
            grid: np.ndarray, shape (8, 10)

        注意：
            grid 的访问方式是 grid[y, x]
            但环境 tile 坐标约定是 (x, y)
        """
        logits = self.predict_logits(pixel_obs)
        grid = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return grid

    @torch.no_grad()
    def predict_grid_with_confidence(self, pixel_obs: np.ndarray) -> dict:
        """
        Returns:
            {
                "grid": np.ndarray, shape (8, 10), uint8,
                "confidence": np.ndarray, shape (8, 10), float32,
                "prob": np.ndarray, shape (C, 8, 10), float32,
            }
        """
        logits = self.predict_logits(pixel_obs)

        prob = torch.softmax(logits, dim=1)      # (1, C, 8, 10)
        confidence, pred = prob.max(dim=1)      # (1, 8, 10)

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
        用某个类别的概率图找最可能的位置。

        Args:
            prob: shape (C, 8, 10)
            class_id: 类别编号
            threshold: 置信度低于该值则返回 [-1, -1]

        Returns:
            pos_xy: np.ndarray, shape (2,), [x, y]
            conf: float

        说明：
            prob[class_id] 的 shape 是 (8, 10)，访问方式是 [y, x]
            但返回给外部的 tile 坐标按环境约定是 [x, y]
        """
        if class_id < 0 or class_id >= prob.shape[0]:
            return np.array([-1, -1], dtype=np.int32), 0.0

        class_prob = prob[class_id]  # shape: (8, 10), index as [y, x]

        flat_idx = int(class_prob.argmax())
        y, x = np.unravel_index(flat_idx, class_prob.shape)

        conf = float(class_prob[y, x])

        if conf < threshold:
            return np.array([-1, -1], dtype=np.int32), conf

        # 环境坐标约定是 [x, y]
        return np.array([x, y], dtype=np.int32), conf

    def _find_all_objects_from_grid_xy(
        self,
        grid: np.ndarray,
        class_id: int,
        max_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        从 argmax grid 中找所有 class_id 的位置。

        Args:
            grid: shape (8, 10)，访问方式 grid[y, x]
            class_id: 类别编号
            max_count: 最多返回数量

        Returns:
            positions_xy: shape (max_count, 2)，每个位置是 [x, y]
            active_mask: shape (max_count,)
        """
        positions_yx = np.argwhere(grid == class_id)

        positions_xy = np.full((max_count, 2), -1, dtype=np.int32)
        active_mask = np.zeros((max_count,), dtype=bool)

        for idx, (y, x) in enumerate(positions_yx[:max_count]):
            positions_xy[idx] = np.array([x, y], dtype=np.int32)
            active_mask[idx] = True

        return positions_xy, active_mask

    def predict_state(
        self,
        pixel_obs: np.ndarray,
        player_threshold: float = 0.0,
        max_monsters: int = 8,
    ) -> dict:
        """
        Args:
            pixel_obs: RGB image, shape (128, 160, 3), dtype uint8
            player_threshold: 玩家置信度低于该值则返回 [-1, -1]
            max_monsters: 最多返回怪物数量

        Returns:
            {
                "grid": np.ndarray, shape (8, 10),
                "confidence": np.ndarray, shape (8, 10),
                "player_tile": np.ndarray, shape (2,), [x, y],
                "player_confidence": float,
                "monsters_tile": np.ndarray, shape (max_monsters, 2), each [x, y],
                "monsters_active_mask": np.ndarray, shape (max_monsters,), bool,
            }
        """
        result = self.predict_grid_with_confidence(pixel_obs)

        grid = result["grid"]
        confidence = result["confidence"]
        prob = result["prob"]

        # 玩家只有一个，用 PLAYER_ID 的概率图找最大概率位置，更稳定
        player_tile, player_confidence = self._find_single_object_by_prob_xy(
            prob=prob,
            class_id=PLAYER_ID,
            threshold=player_threshold,
        )

        # 怪物可能有多个，用 argmax grid 找所有 monster
        monsters_tile, monsters_active_mask = self._find_all_objects_from_grid_xy(
            grid=grid,
            class_id=MONSTER_ID,
            max_count=max_monsters,
        )

        return {
            "grid": grid,
            "confidence": confidence,
            "player_tile": player_tile,                  # [x, y]
            "player_confidence": player_confidence,
            "monsters_tile": monsters_tile,              # each [x, y]
            "monsters_active_mask": monsters_active_mask,
        }