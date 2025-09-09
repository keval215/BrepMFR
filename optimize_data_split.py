#!/usr/bin/env python3
"""
Optimize data split for BrepMFR training
Analyzes current data and suggests optimal train/val split
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
import shutil

def analyze_current_split():
    """Analyze current train/val split"""
    
    # Read current splits
    train_files = []
    val_files = []
    
    with open("data/train.txt", "r") as f:
        train_files = [x.strip() for x in f.readlines()]
    
    with open("data/val.txt", "r") as f:
        val_files = [x.strip() for x in f.readlines()]
    
    print(f"Current split:")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Total files: {len(train_files) + len(val_files)}")
    print(f"Validation ratio: {len(val_files)/(len(train_files) + len(val_files)):.1%}")
    
    # Analyze feature distribution in training data
    train_features = []
    for file_id in train_files[:10]:  # Sample first 10 files
        json_path = Path(f"data/train/{file_id}.json")
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
                train_features.extend(data["labels"])
    
    feature_dist = Counter(train_features)
    print(f"\nFeature distribution in training (sample):")
    for feature, count in sorted(feature_dist.items()):
        print(f"Feature {feature}: {count} faces")
    
    return len(train_files), len(val_files)

def suggest_optimal_split(total_files, target_val_ratio=0.2):
    """Suggest optimal train/val split"""
    
    optimal_val_size = int(total_files * target_val_ratio)
    optimal_train_size = total_files - optimal_val_size
    
    print(f"\nSuggested optimal split ({target_val_ratio:.0%} validation):")
    print(f"Training files: {optimal_train_size}")
    print(f"Validation files: {optimal_val_size}")
    print(f"Validation ratio: {optimal_val_size/total_files:.1%}")
    
    return optimal_train_size, optimal_val_size

def main():
    print("=== BrepMFR Data Split Analysis ===")
    
    train_count, val_count = analyze_current_split()
    total_files = train_count + val_count
    
    # Current validation ratio
    current_ratio = val_count / total_files
    
    print(f"\n=== Analysis ===")
    if current_ratio < 0.15:
        print("⚠️  Validation set is quite small (< 15%)")
        print("   Consider increasing to 20-25% for better validation")
    elif current_ratio < 0.20:
        print("✅ Validation set size is acceptable (15-20%)")
        print("   Could be increased slightly for better reliability")
    else:
        print("✅ Validation set size is good (> 20%)")
    
    # Suggestions for different scenarios
    print(f"\n=== Recommendations ===")
    print(f"Current setup ({val_count} val files): Good for quick experiments")
    
    if total_files >= 150:
        suggest_optimal_split(total_files, 0.20)
        suggest_optimal_split(total_files, 0.25)
    
    print(f"\n=== For SageMaker Training ===")
    print(f"Your current {val_count} validation files are sufficient for:")
    print("- Proof of concept training")
    print("- Model development and debugging") 
    print("- Small-scale experiments")
    print(f"\nRecommendation: Keep current split for now, increase later if you get more data")

if __name__ == "__main__":
    main()