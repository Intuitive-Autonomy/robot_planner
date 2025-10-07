#!/usr/bin/env python3
"""
Preprocess augmented CSV data into efficient NumPy format
Creates separate NPZ files for train/val/test splits
"""

import os
import numpy as np
from tqdm import tqdm
import argparse
from utils import AugmentedDataProcessor


def preprocess_split_data(data_root: str,
                          output_dir: str,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1,
                          test_ratio: float = 0.1,
                          random_seed: int = 42):
    """
    Load CSV files and save as separate train/val/test .npz files

    Args:
        data_root: Root directory containing augmented CSV files
        output_dir: Directory to save the NPZ files
        train_ratio: Ratio for training split
        val_ratio: Ratio for validation split
        test_ratio: Ratio for test split
        random_seed: Random seed for reproducible splits
    """
    processor = AugmentedDataProcessor(data_root)
    file_groups = processor.file_groups
    base_names = processor.get_base_names()

    print(f"Found {len(base_names)} base recordings with {sum(len(g) for g in file_groups.values())} total files")

    # Create train/val/test splits (same logic as in utils.py)
    np.random.seed(random_seed)
    shuffled_bases = np.random.permutation(base_names).tolist()

    n_total = len(shuffled_bases)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_bases = shuffled_bases[:n_train]
    val_bases = shuffled_bases[n_train:n_train+n_val]
    test_bases = shuffled_bases[n_train+n_val:]

    train_files = [f for base in train_bases for f in file_groups[base]]
    val_files = [f for base in val_bases for f in file_groups[base]]
    test_files = [f for base in test_bases for f in file_groups[base]]

    print(f"\nData Split:")
    print(f"  Train: {len(train_bases)} bases ({len(train_files)} files)")
    print(f"  Val:   {len(val_bases)} bases ({len(val_files)} files)")
    print(f"  Test:  {len(test_bases)} bases ({len(test_files)} files)")

    # Process each split
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    os.makedirs(output_dir, exist_ok=True)

    for split_name, file_list in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} split ({len(file_list)} files)...")
        print(f"{'='*60}")

        split_data = {}

        for file_path in tqdm(file_list, desc=f"{split_name}", ncols=100):
            try:
                # Extract relative path as key
                rel_path = os.path.relpath(file_path, data_root)

                # Load and process the file
                inputs, targets = processor.load_trajectory_file(file_path)

                # Store with relative path as key
                split_data[f"{rel_path}_inputs"] = inputs
                split_data[f"{rel_path}_targets"] = targets

            except Exception as e:
                print(f"\nWarning: Failed to load {file_path}: {e}")
                continue

        # Save split to .npz
        output_path = os.path.join(output_dir, f"preprocessed_{split_name}.npz")
        print(f"Saving {split_name} data to {output_path}...")
        np.savez_compressed(output_path, **split_data)

        # Calculate file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Done! Saved {len(split_data)//2} trajectories ({file_size_mb:.1f} MB)")

    print(f"\n{'='*60}")
    print("All splits processed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"  - preprocessed_train.npz")
    print(f"  - preprocessed_val.npz")
    print(f"  - preprocessed_test.npz")


def verify_preprocessed_data(npz_path: str):
    """Verify the preprocessed data can be loaded correctly"""
    print(f"\nVerifying {npz_path}...")

    data = np.load(npz_path)
    print(f"Loaded {len(data.files)//2} trajectories")

    # Show sample
    sample_keys = list(data.keys())[:4]
    for key in sample_keys:
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    data.close()
    print("Verification complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess augmented CSV data to NPZ format (train/val/test splits)')
    parser.add_argument('--data_root', type=str, default='/mnt/fsx/fsx/global_csv',
                       help='Root directory containing augmented CSV files')
    parser.add_argument('--output_dir', type=str, default='/mnt/fsx/fsx/global_csv/preprocessed',
                       help='Output directory for preprocessed NPZ files')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train split ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test split ratio (default: 0.1)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the preprocessed data after creation')

    args = parser.parse_args()

    # Preprocess
    preprocess_split_data(
        args.data_root,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.random_seed
    )

    # Verify if requested
    if args.verify:
        for split in ['train', 'val', 'test']:
            npz_path = os.path.join(args.output_dir, f"preprocessed_{split}.npz")
            if os.path.exists(npz_path):
                verify_preprocessed_data(npz_path)
