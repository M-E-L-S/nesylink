from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from nesylink.env import make_env

TASK_IDS = [
    "mathematical_logic/task_1",
    "mathematical_logic/task_2",
    "mathematical_logic/task_3",
    "mathematical_logic/task_4",
    "mathematical_logic/task_5",
]

class PixelGridDataset(Dataset):
    def __init__(self, path: str | Path):
        data = np.load(path)
        self.pixels = data["pixels"]
        self.grids = data["grids"]

    def __len__(self) -> int:
        return len(self.pixels)

    def __getitem__(self, index: int):
        pixel = self.pixels[index].astype(np.float32) / 255.0
        grid = self.grids[index].astype(np.int64)

        pixel = torch.from_numpy(pixel).permute(2, 0, 1)
        grid = torch.from_numpy(grid)

        return pixel, grid

def collect_dataset(
    output_path: str | Path,
    task_ids: Iterable[str] = TASK_IDS,
    episodes_per_task: int = 20,
    steps_per_episode: int = 200,
    seed: int = 0,
) -> None:
    """
    Collect paired pixel observations and grid labels.

    For each task, this creates two synchronized environments:
    one returns pixels, the other returns structured grid labels.
    They are reset with the same seed and stepped with the same actions.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pixels = []
    grids = []

    rng = np.random.default_rng(seed)

    for task_id in task_ids:
        print(f"Collecting task: {task_id}")

        env_pixels = make_env(
            task_id=task_id,
            observation_mode="pixels",
            control_mode="pixel",
        )
        env_full = make_env(
            task_id=task_id,
            observation_mode="full",
            control_mode="pixel",
        )

        for episode in range(episodes_per_task):
            episode_seed = seed + episode
            obs_pixel, _ = env_pixels.reset(seed=episode_seed)
            obs_full, _ = env_full.reset(seed=episode_seed)

            pixels.append(obs_pixel)
            grids.append(obs_full["grid"])

            for _ in range(steps_per_episode):
                action = int(rng.integers(env_pixels.action_space.n))

                obs_pixel, _, terminated_p, truncated_p, _ = env_pixels.step(action)
                obs_full, _, terminated_f, truncated_f, _ = env_full.step(action)

                pixels.append(obs_pixel)
                grids.append(obs_full["grid"])

                if terminated_p or truncated_p or terminated_f or truncated_f:
                    break

        env_pixels.close()
        env_full.close()

    pixels_array = np.asarray(pixels, dtype=np.uint8)
    grids_array = np.asarray(grids, dtype=np.uint8)

    np.savez_compressed(
        output_path,
        pixels=pixels_array,
        grids=grids_array,
    )

    print(f"Saved dataset: {output_path}")
    print(f"pixels: {pixels_array.shape}, grids: {grids_array.shape}")