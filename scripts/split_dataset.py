import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split raw dataset into train/val/test directories.")
    parser.add_argument("--source", default="data/raw", help="Directory holding the raw class folders.")
    parser.add_argument("--destination", default="data/processed", help="Destination root for split datasets.")
    parser.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15], help="Ratios for train/val/test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def copy_files(files, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dest_dir / src.name)


def main():
    args = parse_args()
    random.seed(args.seed)
    source_dir = Path(args.source)
    dest_dir = Path(args.destination)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    train_ratio, val_ratio, test_ratio = args.split
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*"))
        random.shuffle(images)
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        copy_files(train_files, dest_dir / "train" / class_dir.name)
        copy_files(val_files, dest_dir / "val" / class_dir.name)
        copy_files(test_files, dest_dir / "test" / class_dir.name)

    print(f"Dataset split complete. Output saved to {dest_dir}")


if __name__ == "__main__":
    main()


