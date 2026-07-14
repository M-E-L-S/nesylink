import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.pixel2state.dataset import PixelGridDataset, collect_dataset
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

def augment_pixel_obs(pixels: torch.Tensor) -> torch.Tensor:
    """
    pixels: torch.Tensor, shape (3, 128, 160), values usually in [0, 1]

    Robust augmentations for evaluator variants:
    - grayscale
    - dark / bright
    - high contrast
    - inverted
    - channel dropout / shuffle
    - mild noise
    - posterization-like quantization
    """
    x = pixels.float().clamp(0.0, 1.0)

    mode = torch.randint(0, 10, (1,)).item()

    if mode == 0:
        # original
        pass

    elif mode == 1:
        # grayscale
        gray = 0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]
        x = gray.repeat(3, 1, 1)

    elif mode == 2:
        # dark
        scale = torch.empty(1).uniform_(0.35, 0.75).item()
        x = x * scale

    elif mode == 3:
        # bright
        scale = torch.empty(1).uniform_(1.15, 1.6).item()
        bias = torch.empty(1).uniform_(0.02, 0.12).item()
        x = x * scale + bias

    elif mode == 4:
        # high contrast
        threshold = torch.empty(1).uniform_(0.35, 0.65).item()
        x = torch.where(x > threshold, torch.ones_like(x), torch.zeros_like(x))

    elif mode == 5:
        # inverted
        x = 1.0 - x

    elif mode == 6:
        # channel shuffle
        perm = torch.randperm(3)
        x = x[perm]

    elif mode == 7:
        # channel dropout, forces shape/brightness cues instead of exact color
        channel = torch.randint(0, 3, (1,)).item()
        x[channel] = x[channel] * torch.empty(1).uniform_(0.0, 0.25).item()

    elif mode == 8:
        # mild gaussian noise
        noise_scale = torch.empty(1).uniform_(0.01, 0.06).item()
        x = x + torch.randn_like(x) * noise_scale

    elif mode == 9:
        # posterization / redraw-like flat colors
        levels = torch.randint(2, 6, (1,)).item()
        x = torch.round(x * (levels - 1)) / float(levels - 1)

    return x.clamp(0.0, 1.0)

def normalize_pixels_batch(pixels: torch.Tensor) -> torch.Tensor:
    """
    Match PixelToStatePredictor._preprocess() in infer.py.
    """
    pixels = pixels.float()

    mean = pixels.mean(dim=(2, 3), keepdim=True)
    std = pixels.std(dim=(2, 3), keepdim=True)

    pixels = (pixels - mean) / (std + 1e-6)
    pixels = pixels.clamp(-1.0, 1.0)

    return pixels

class AugmentedDataset:
    def __init__(self, dataset, augment: bool = True):
        self.dataset = dataset
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pixels, grids = self.dataset[idx]

        if self.augment:
            pixels = augment_pixel_obs(pixels)

        return pixels, grids

def train(
    data_path: str,
    output_path: str,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
    num_tile_classes: int = 12,
    device: str | None = None,
    augment: bool = True,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    base_dataset = PixelGridDataset(data_path)

    val_size = max(1, int(len(base_dataset) * 0.1))
    train_size = len(base_dataset) - val_size

    train_base_set, val_set = random_split(
        base_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    train_set = AugmentedDataset(train_base_set, augment=augment)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = make_model(num_tile_classes=num_tile_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_player_acc = 0.0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_tiles = 0
        correct_tiles = 0

        for pixels, grids in train_loader:
            pixels = normalize_pixels_batch(pixels.to(device))
            grids = grids.to(device).long()

            logits = model(pixels)
            loss = criterion(logits, grids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pixels.size(0)

            pred = logits.argmax(dim=1)
            correct_tiles += (pred == grids).sum().item()
            total_tiles += grids.numel()

        train_loss = total_loss / len(train_set)
        train_acc = correct_tiles / total_tiles

        val_loss, val_acc = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        player_metrics = evaluate_player_tile_loader(
            model=model,
            loader=val_loader,
            device=device,
        )

        player_acc = player_metrics["player_acc_by_prob"]

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"player_acc_by_prob={player_metrics['player_acc_by_prob']:.4f} "
            f"player_acc_by_argmax_grid={player_metrics['player_acc_by_argmax_grid']:.4f} "
            f"avg_player_conf={player_metrics['avg_player_conf']:.4f} "
            f"missing_true_player={player_metrics['missing_true_player']}"
        )

        # Keep the old tile-accuracy save behavior, but record player-specific
        # metrics so we can tell whether the model learned player_tile.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_player_acc = player_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_tile_classes": num_tile_classes,
                    "val_acc": val_acc,
                    "player_acc_by_prob": player_metrics["player_acc_by_prob"],
                    "player_acc_by_argmax_grid": player_metrics[
                        "player_acc_by_argmax_grid"
                    ],
                    "avg_player_conf": player_metrics["avg_player_conf"],
                    "epoch": epoch,
                    "input_normalization": "per_image_mean_std_clamp",
                    "color_augmentation": bool(augment),
                    "augmentation_version": "color_channel_noise_posterize_v2",
                },
                output_path,
            )

            print(f"saved best model to {output_path}")

    print(f"best_val_acc={best_val_acc:.4f}")
    print(f"best_player_acc_by_prob_at_saved_epoch={best_player_acc:.4f}")

@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_tiles = 0
    correct_tiles = 0

    for pixels, grids in loader:
        pixels = normalize_pixels_batch(pixels.to(device))
        grids = grids.to(device).long()

        logits = model(pixels)
        loss = criterion(logits, grids)

        total_loss += loss.item() * pixels.size(0)

        pred = logits.argmax(dim=1)
        correct_tiles += (pred == grids).sum().item()
        total_tiles += grids.numel()

    return total_loss / len(loader.dataset), correct_tiles / total_tiles

@torch.no_grad()
def evaluate_player_tile_loader(model, loader, device):
    """
    Evaluate whether the model can locate the player tile.

    This mirrors infer.py behavior:
    infer.py uses PLAYER_ID probability map and takes the tile with max PLAYER_ID
    probability.
    """
    model.eval()

    total = 0
    correct_by_player_prob = 0
    correct_by_argmax_grid = 0
    missing_true_player = 0
    avg_player_conf = 0.0

    for pixels, grids in loader:
        pixels = normalize_pixels_batch(pixels.to(device))
        grids = grids.to(device).long()

        logits = model(pixels)
        prob = torch.softmax(logits, dim=1)
        pred_grid = logits.argmax(dim=1)

        batch_size = grids.size(0)

        for i in range(batch_size):
            true_pos = torch.nonzero(grids[i] == PLAYER_ID, as_tuple=False)

            if true_pos.numel() == 0:
                missing_true_player += 1
                continue

            true_y = int(true_pos[0, 0].item())
            true_x = int(true_pos[0, 1].item())

            player_prob = prob[i, PLAYER_ID]
            flat_idx = int(player_prob.reshape(-1).argmax().item())
            pred_y = flat_idx // player_prob.shape[1]
            pred_x = flat_idx % player_prob.shape[1]

            avg_player_conf += float(player_prob[pred_y, pred_x].item())

            if pred_x == true_x and pred_y == true_y:
                correct_by_player_prob += 1

            argmax_pos = torch.nonzero(pred_grid[i] == PLAYER_ID, as_tuple=False)
            if argmax_pos.numel() > 0:
                argmax_y = int(argmax_pos[0, 0].item())
                argmax_x = int(argmax_pos[0, 1].item())

                if argmax_x == true_x and argmax_y == true_y:
                    correct_by_argmax_grid += 1

            total += 1

    if total == 0:
        return {
            "player_acc_by_prob": 0.0,
            "player_acc_by_argmax_grid": 0.0,
            "avg_player_conf": 0.0,
            "missing_true_player": missing_true_player,
        }

    return {
        "player_acc_by_prob": correct_by_player_prob / total,
        "player_acc_by_argmax_grid": correct_by_argmax_grid / total,
        "avg_player_conf": avg_player_conf / total,
        "missing_true_player": missing_true_player,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/pixel2state/train.npz")
    parser.add_argument("--output", default="models/pixel2state/best.pt")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-tile-classes", type=int, default=12)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable color augmentation.",
    )

    args = parser.parse_args()

    if args.collect or not Path(args.data).exists():
        collect_dataset(
            output_path=args.data,
            episodes_per_task=args.episodes_per_task,
            steps_per_episode=args.steps_per_episode,
        )

    train(
        data_path=args.data,
        output_path=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_tile_classes=args.num_tile_classes,
        augment=not args.no_augment,
    )

if __name__ == "__main__":
    main()