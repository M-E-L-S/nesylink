import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from experiments.pixel2state.dataset import PixelGridDataset, collect_dataset
from experiments.pixel2state.model import make_model

def train(
    data_path: str,
    output_path: str,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PixelGridDataset(data_path)

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

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

    model = make_model(num_tile_classes=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tiles = 0
        correct_tiles = 0

        for pixels, grids in train_loader:
            pixels = pixels.to(device)
            grids = grids.to(device)

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

        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_tile_classes": 12,
                    "val_acc": val_acc,
                },
                output_path,
            )
            print(f"saved best model to {output_path}")

    print(f"best_val_acc={best_val_acc:.4f}")

@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_tiles = 0
    correct_tiles = 0

    for pixels, grids in loader:
        pixels = pixels.to(device)
        grids = grids.to(device)

        logits = model(pixels)
        loss = criterion(logits, grids)

        total_loss += loss.item() * pixels.size(0)

        pred = logits.argmax(dim=1)
        correct_tiles += (pred == grids).sum().item()
        total_tiles += grids.numel()

    return total_loss / len(loader.dataset), correct_tiles / total_tiles

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
    )

if __name__ == "__main__":
    main()