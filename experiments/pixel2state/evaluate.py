import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.pixel2state.dataset import PixelGridDataset
from experiments.pixel2state.model import make_model

@torch.no_grad()
def evaluate(model_path: str, data_path: str, batch_size: int = 64):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PixelGridDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(model_path, map_location=device)
    num_tile_classes = checkpoint.get("num_tile_classes", 12)

    model = make_model(num_tile_classes=num_tile_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    total_tiles = 0
    correct_tiles = 0

    per_class_total = torch.zeros(num_tile_classes, dtype=torch.long)
    per_class_correct = torch.zeros(num_tile_classes, dtype=torch.long)

    for pixels, grids in loader:
        pixels = pixels.to(device)
        grids = grids.to(device)

        logits = model(pixels)
        pred = logits.argmax(dim=1)

        correct = pred == grids

        correct_tiles += correct.sum().item()
        total_tiles += grids.numel()

        for tile_id in range(num_tile_classes):
            mask = grids == tile_id
            per_class_total[tile_id] += mask.sum().cpu()
            per_class_correct[tile_id] += (correct & mask).sum().cpu()

    overall_acc = correct_tiles / total_tiles

    class_acc = {}
    for tile_id in range(num_tile_classes):
        total = int(per_class_total[tile_id])
        correct = int(per_class_correct[tile_id])
        class_acc[str(tile_id)] = None if total == 0 else correct / total

    result = {
        "overall_tile_accuracy": overall_acc,
        "per_class_tile_accuracy": class_acc,
        "num_samples": len(dataset),
        "num_tile_classes": num_tile_classes,
    }

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pixel2state/best.pt")
    parser.add_argument("--data", default="data/pixel2state/train.npz")
    parser.add_argument("--output", default="models/pixel2state/eval_results.json")
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    result = evaluate(
        model_path=args.model,
        data_path=args.data,
        batch_size=args.batch_size,
    )

    print(json.dumps(result, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()