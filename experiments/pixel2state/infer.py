from pathlib import Path

import numpy as np
import torch

from experiments.pixel2state.model import make_model

class PixelToStatePredictor:
    def __init__(self, model_path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = make_model(
            num_tile_classes=checkpoint.get("num_tile_classes", 12)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_grid(self, pixel_obs: np.ndarray) -> np.ndarray:
        """
        Args:
            pixel_obs: uint8 RGB array, shape (128, 160, 3)

        Returns:
            grid: uint8 array, shape (8, 10)
        """
        if pixel_obs.shape != (128, 160, 3):
            raise ValueError(f"expected pixel_obs shape (128, 160, 3), got {pixel_obs.shape}")

        x = pixel_obs.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

        logits = self.model(x)
        grid = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        return grid

    def predict_state(self, pixel_obs: np.ndarray) -> dict:
        grid = self.predict_grid(pixel_obs)

        player_positions = np.argwhere(grid == 2)
        if len(player_positions) > 0:
            y, x = player_positions[0]
            player_tile = np.array([x, y], dtype=np.int32)
        else:
            player_tile = np.array([-1, -1], dtype=np.int32)

        monster_positions_yx = np.argwhere(grid == 3)
        monsters_tile = np.full((8, 2), -1, dtype=np.int32)
        monsters_active_mask = np.zeros((8,), dtype=bool)

        for idx, (y, x) in enumerate(monster_positions_yx[:8]):
            monsters_tile[idx] = np.array([x, y], dtype=np.int32)
            monsters_active_mask[idx] = True

        return {
            "grid": grid,
            "player_tile": player_tile,
            "monsters_tile": monsters_tile,
            "monsters_active_mask": monsters_active_mask,
        }