#!/usr/bin/env python3
"""
Script to visualize predicted features on CAD models
"""
import numpy as np
import json
from pathlib import Path

def load_predictions(feature_file):
    """Load feature predictions from output file"""
    with open(feature_file, 'r') as f:
        predictions = [int(x.strip()) for x in f.readlines() if x.strip()]
    return predictions

def load_ground_truth(json_file):
    """Load ground truth labels from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['labels']

def analyze_features(predictions, ground_truth=None):
    """Analyze feature distribution"""
    unique_features, counts = np.unique(predictions, return_counts=True)
    
    print("Feature Analysis:")
    print("================")
    print(f"Total faces: {len(predictions)}")
    for feature, count in zip(unique_features, counts):
        percentage = (count / len(predictions)) * 100
        print(f"Feature {feature:2d}: {count:3d} faces ({percentage:5.1f}%)")
    
    if ground_truth:
        print(f"\nGround Truth Analysis:")
        print("======================")
        print(f"Ground truth faces: {len(ground_truth)}")
        print(f"Prediction faces: {len(predictions)}")
        
        if len(ground_truth) != len(predictions):
            print("WARNING: Mismatch in number of faces between prediction and ground truth!")
            print("This might indicate different graph structures or processing.")
            
            # Try to compare the overlapping portion
            min_len = min(len(predictions), len(ground_truth))
            pred_subset = predictions[:min_len]
            gt_subset = ground_truth[:min_len]
            
            correct = sum(p == g for p, g in zip(pred_subset, gt_subset))
            accuracy = (correct / min_len) * 100
            print(f"Accuracy on first {min_len} faces: {correct}/{min_len} ({accuracy:.1f}%)")
        else:
            correct = sum(p == g for p, g in zip(predictions, ground_truth))
            accuracy = (correct / len(predictions)) * 100
            print(f"Overall accuracy: {correct}/{len(predictions)} ({accuracy:.1f}%)")

def create_feature_mapping():
    """Official BrepMFR feature mapping (25 classes: 0-24)"""
    feature_map = {
        0: "Base/Stock Material",
        1: "Rectangular Through Slot",
        2: "Triangular Through Slot", 
        3: "Rectangular Passage",
        4: "Triangular Passage",
        5: "6-Sided Passage",
        6: "Rectangular Through Step",
        7: "2-Sided Through Step",
        8: "Slanted Through Step",
        9: "Rectangular Blind Step",
        10: "Triangular Blind Step",
        11: "Rectangular Blind Slot",
        12: "Rectangular Pocket",
        13: "Triangular Pocket",
        14: "6-Sides Pocket",
        15: "Chamfer",
        16: "Circular Through Slot",
        17: "Through Hole",
        18: "Circular Blind Step",
        19: "Horizontal Circular End Blind Slot",
        20: "Vertical Circular End Blind Slot",
        21: "Circular End Pocket",
        22: "O-ring",
        23: "Blind Hole",
        24: "Fillet"
    }
    return feature_map

if __name__ == "__main__":
    # Analyze your specific file
    feature_file = "test_output/features/feature_151.txt"
    json_file = "data/test/00000151.json"  # If it exists
    
    predictions = load_predictions(feature_file)
    
    # Try to load ground truth if available
    ground_truth = None
    if Path(json_file).exists():
        ground_truth = load_ground_truth(json_file)
    
    analyze_features(predictions, ground_truth)
    
    print("\nFeature Mapping (estimated):")
    print("============================")
    feature_map = create_feature_mapping()
    unique_features = np.unique(predictions)
    
    for feature in unique_features:
        name = feature_map.get(feature, "Unknown Feature")
        print(f"Feature {feature:2d}: {name}")