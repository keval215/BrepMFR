# -*- coding: utf-8 -*-
import argparse
import pathlib
import time
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from argparse import Namespace

from data.dataset import CADSynth
from models.brepseg_model import BrepSeg
from models.modules.utils.macro import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main(checkpoint_manager=None):
    """Main training function that can accept checkpoint_manager"""
    
    parser = argparse.ArgumentParser("BrepMFR Network model")
    parser.add_argument("traintest", choices=("train", "test"), default="train", help="Whether to train or test")
    parser.add_argument("--num_classes", type=int, default=25, help="Number of features")
    parser.add_argument("--dataset", choices=("cadsynth", "transfer"), default="cadsynth", help="Dataset to train on")
    parser.add_argument("--dataset_path", type=str, default="/opt/ml/input/data", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for the dataloader.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file to load weights from for testing",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="BrepMFR",
        help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
    )
    
    # Transformer module default parameters
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention_dropout", type=float, default=0.3)
    parser.add_argument("--act-dropout", type=float, default=0.3)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dim_node", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_layers_encode", type=int, default=8)
    
    # Add distributed training arguments (avoid conflicts with Lightning)
    # REMOVED: parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    # Don't add --gpus here, let Lightning handle it
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    
    # Add Lightning's arguments FIRST, then we can override defaults
    parser = Trainer.add_argparse_args(parser)
    
    # Parse known args to handle SageMaker hyperparameters
    args, unknown = parser.parse_known_args()
    
    # Set Lightning defaults after parsing (if not already set)
    if not hasattr(args, 'max_epochs') or args.max_epochs is None:
        args.max_epochs = 100
    
    # Override with SageMaker hyperparameters if they exist
    sm_hyperparams = {}
    for key, value in os.environ.items():
        if key.startswith('SM_HP_'):
            param_name = key[6:].lower()  # Remove 'SM_HP_' prefix
            try:
                # Try to convert to appropriate type
                if param_name in ['batch_size', 'num_workers', 'num_classes', 'max_epochs', 'gpus', 'nodes']:
                    sm_hyperparams[param_name] = int(value)
                elif param_name in ['dropout', 'attention_dropout', 'act_dropout']:
                    sm_hyperparams[param_name] = float(value)
                else:
                    sm_hyperparams[param_name] = value
            except ValueError:
                sm_hyperparams[param_name] = value
    
    # Update args with SageMaker hyperparameters
    for key, value in sm_hyperparams.items():
        if hasattr(args, key):
            setattr(args, key, value)
            print(f"Updated {key} = {value} from SageMaker hyperparameters")
    
    # Detect distributed training environment
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"Distributed training info: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    
    # Configure for single instance with multiple GPUs
    num_gpus_available = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus_available}")
    
    if num_gpus_available > 1:
        # Multi-GPU on single node
        print(f"Multi-GPU setup: 1 node, {num_gpus_available} GPUs")
        
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        devices = num_gpus_available  # Use all available GPUs
        num_nodes = 1
    else:
        # Single GPU or CPU
        strategy = "auto"
        devices = 1 if torch.cuda.is_available() else None
        num_nodes = 1
        print("Single GPU/CPU setup")
    
    results_path = pathlib.Path("/opt/ml/model/results").joinpath(args.experiment_name)
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    
    month_day = time.strftime("%m%d")
    hour_min_second = time.strftime("%H%MS")
    
    # Configure checkpointing
    checkpoint_dir = "/opt/ml/checkpoints" if checkpoint_manager else str(results_path.joinpath(month_day, hour_min_second))
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Changed from "eval_loss" to "val_loss"
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        save_last=True,
        mode="min",
        save_on_train_epoch_end=False,  # Save on validation end
        every_n_epochs=5,  # Save every 5 epochs
    )
    
    # Configure trainer for distributed training
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger(
            "/opt/ml/model/logs", name=args.experiment_name, version=f"{month_day}_{hour_min_second}"
        ),
        gradient_clip_val=1.0,
        precision=16,  # Mixed precision for better performance
        sync_batchnorm=True if world_size > 1 else False,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=50,
        val_check_interval=1.0,  # Validate every epoch
        check_val_every_n_epoch=1,
    )
    
    if args.dataset == "cadsynth":
        Dataset = CADSynth
    else:
        raise ValueError("Unsupported dataset")
    
    if args.traintest == "train":
        print(f"""
    -----------------------------------------------------------------------------------
    B-rep model feature recognition - DISTRIBUTED TRAINING
    -----------------------------------------------------------------------------------
    Configuration:
      - Nodes: {num_nodes}
      - GPUs per node: {devices if isinstance(devices, int) else len(devices) if devices else 0}
      - Total GPUs: {num_gpus_available if 'num_gpus_available' in locals() else devices}
      - Batch size: {args.batch_size}
      - Max epochs: {args.max_epochs}
      - Strategy: {strategy}
    
    Logs written to /opt/ml/model/logs/{args.experiment_name}/{month_day}_{hour_min_second}
    Checkpoints written to {checkpoint_dir}
    -----------------------------------------------------------------------------------
        """)
        
        model = BrepSeg(args)
        
        # Verify data paths
        train_path = os.path.join(args.dataset_path, "train")
        val_path = os.path.join(args.dataset_path, "validation")  # Note: might be "val" or "validation"
        
        if not os.path.exists(train_path):
            train_path = os.path.join(args.dataset_path, "train")
            print(f"Training data path: {train_path}")
            print(f"Training data exists: {os.path.exists(train_path)}")
            if os.path.exists(args.dataset_path):
                print(f"Contents of {args.dataset_path}: {os.listdir(args.dataset_path)}")
        
        if not os.path.exists(val_path):
            val_path = os.path.join(args.dataset_path, "val")
            print(f"Validation data path: {val_path}")
            print(f"Validation data exists: {os.path.exists(val_path)}")
        
        try:
            train_data = Dataset(root_dir=args.dataset_path, split="train", random_rotate=True, num_class=args.num_classes)
            val_data = Dataset(root_dir=args.dataset_path, split="val", random_rotate=False, num_class=args.num_classes)
            
            train_loader = train_data.get_dataloader(
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
            )
            val_loader = val_data.get_dataloader(
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
            )
            
            print(f"Training batches: {len(train_loader)}")
            print(f"Validation batches: {len(val_loader)}")
            
            # Start training
            trainer.fit(model, train_loader, val_loader)
            
            print("Training completed successfully!")
            return {"status": "success", "best_model_path": checkpoint_callback.best_model_path}
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
        
    else:
        # Test mode
        assert args.checkpoint is not None, "Expected the --checkpoint argument to be provided"
        
        print("--- Loading checkpoint for testing ---")
        
        checkpoint_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        hparams_dict = checkpoint_data['hyper_parameters']
        
        merged_dict = vars(args).copy()
        for key, value in hparams_dict.items():
            if key not in merged_dict:
                merged_dict[key] = value
        
        hparams_obj = Namespace(**merged_dict)
        model = BrepSeg(hparams_obj)
        model.load_state_dict(checkpoint_data['state_dict'])
        
        print("--- Checkpoint loaded successfully ---")
        
        output_dir = os.path.join("/opt/ml/model", "test_output", "features")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        os.environ['FEATURE_OUTPUT_DIR'] = output_dir

        test_data = Dataset(root_dir=args.dataset_path, split="test", random_rotate=False, num_class=args.num_classes)
        test_loader = test_data.get_dataloader(
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        
        trainer.test(model, dataloaders=test_loader)
        return {"status": "test_completed"}


if __name__ == '__main__':
    result = main()
    print(f"Final result: {result}")