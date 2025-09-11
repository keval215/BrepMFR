#!/usr/bin/env python3
"""
Simple test script to run inference with trained checkpoint
"""
import os
import torch
import numpy as np
from argparse import Namespace
from tqdm import tqdm

# Set up paths
os.environ['FEATURE_OUTPUT_DIR'] = os.path.join(os.getcwd(), "test_output", "features")

from data.dataset import CADSynth
from models.brepseg_model import BrepSeg

def test_model():
    # Load checkpoint
    checkpoint_path = "./last.ckpt"
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams_dict = checkpoint_data['hyper_parameters']
    
    # Create model arguments with defaults
    args = Namespace(
        dataset_path="data",
        batch_size=8,  # Smaller batch size for small dataset
        num_workers=0,
        num_classes=25,
        # Add default parameters that might be missing
        dropout=0.3,
        attention_dropout=0.3,
        act_dropout=0.3,
        d_model=512,
        dim_node=256,
        n_heads=32,
        n_layers_encode=8,
        **hparams_dict
    )
    
    # Create and load model
    model = BrepSeg(args)
    model.load_state_dict(checkpoint_data['state_dict'])
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.environ['FEATURE_OUTPUT_DIR']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load test data
    test_data = CADSynth(root_dir=args.dataset_path, split="test", random_rotate=False, num_class=args.num_classes)
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    print(f"Test dataset loaded: {len(test_data)} samples, {len(test_loader)} batches")
    
    # Run inference
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            node_emb, graph_emb = model.brep_encoder(batch, last_state_only=True)
            
            # Node classifier
            node_emb = node_emb[0].permute(1, 0, 2)[:, 1:, :]  # Remove global node
            padding_mask = batch["padding_mask"]
            node_pos = torch.where(padding_mask == False)
            node_z = node_emb[node_pos]
            padding_mask_ = ~padding_mask
            
            num_nodes_per_graph = torch.sum(padding_mask_.long(), dim=-1)
            graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
            z = model.attention([node_z, graph_z])
            node_seg = model.classifier(z)
            
            # Get predictions
            preds = torch.argmax(node_seg, dim=-1)
            labels = batch["label_feature"].long()
            
            # Filter known classes
            known_pos = torch.where(labels < model.num_classes)
            labels_ = labels[known_pos]
            preds_ = preds[known_pos]
            
            # Convert to numpy
            labels_np = labels_.cpu().numpy()
            preds_np = preds_.cpu().numpy()
            
            all_preds.extend(preds_np)
            all_labels.extend(labels_np)
            
            # Save feature outputs (similar to test_step)
            n_graph, max_n_node = padding_mask.size()[:2]
            face_feature = -1 * torch.ones([n_graph, max_n_node], device=device, dtype=torch.long)
            face_feature[node_pos] = preds[:]
            out_face_feature = face_feature.cpu().numpy()
            
            for i in range(n_graph):
                end_index = max_n_node - np.sum((out_face_feature[i][:] == -1).astype(np.int32))
                pred_feature = out_face_feature[i][:end_index]
                
                file_name = f"feature_{batch['id'][i].cpu().numpy()}.txt"
                file_path = os.path.join(output_dir, file_name)
                
                with open(file_path, "w") as f:
                    for j in range(end_index):
                        f.write(f"{pred_feature[j]}\n")
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    print(f"\nTest Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Total samples: {len(all_preds)}")
    print(f"Features saved to: {output_dir}")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(model.num_classes):
        class_mask = all_labels == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_preds[class_mask] == all_labels[class_mask])
            print(f"Class {i}: {class_acc:.4f} ({np.sum(class_mask)} samples)")

if __name__ == "__main__":
    test_model()
