# -*- coding: utf-8 -*-
import argparse
import pathlib
import time
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace # <--- VVVV ADD THIS IMPORT VVVV

from data.dataset import CADSynth
from models.brepseg_model import BrepSeg
from models.modules.utils.macro import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':

    parser = argparse.ArgumentParser("BrepMFR Network model")
    parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
    parser.add_argument("--num_classes", type=int, default=25, help="Number of features")
    parser.add_argument("--dataset", choices=("cadsynth", "transfer"), default="cadsynth", help="Dataset to train on")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset")
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
    
    #设置transformer模块的默认参数
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention_dropout", type=float, default=0.3)
    parser.add_argument("--act-dropout", type=float, default=0.3)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dim_node", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_layers_encode", type=int, default=8)
    
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    results_path = (
        pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
    )
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    
    month_day = time.strftime("%m%d")
    hour_min_second = time.strftime("%H%MS")
    checkpoint_callback = ModelCheckpoint(
        monitor="eval_loss",
        dirpath=str(results_path.joinpath(month_day, hour_min_second)),
        filename="best",
        save_top_k=10,
        save_last=True,
    )
    
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger(
            str(results_path), name=month_day, version=hour_min_second,
        ),
        accelerator='gpu',
        devices=1,
        auto_select_gpus=True, 
        gradient_clip_val=1.0
    )
    
    if args.dataset == "cadsynth":
        Dataset = CADSynth
    else:
        raise ValueError("Unsupported dataset")
    
    if args.traintest == "train":
        # Train/val
        print(
            f"""
    -----------------------------------------------------------------------------------
    B-rep model feature recognition
    -----------------------------------------------------------------------------------
    Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}
    
    To monitor the logs, run:
    tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}
    
    The trained model with the best validation loss will be written to:
    results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
    -----------------------------------------------------------------------------------
        """
        )
        model = BrepSeg(args)
    
        train_data = Dataset(root_dir=args.dataset_path, split="train", random_rotate=True, num_class=args.num_classes)
        val_data = Dataset(root_dir=args.dataset_path, split="val", random_rotate=False, num_class=args.num_classes)
        train_loader = train_data.get_dataloader(
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        val_loader = val_data.get_dataloader(
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )     
        trainer.fit(model, train_loader, val_loader)
        
    else:
        # Test
        assert (
            args.checkpoint is not None
        ), "Expected the --checkpoint argument to be provided"
        
        print("--- Manually loading checkpoint for testing ---")
        
        checkpoint_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        # The checkpoint hparams are a dictionary, but the model expects an object.
        # VVVV THIS IS THE FIX VVVV
        hparams_dict = checkpoint_data['hyper_parameters']
        
        # Merge current args with saved hparams, prioritizing current args for testing
        merged_dict = vars(args).copy()  # Start with current command line args
        for key, value in hparams_dict.items():
            if key not in merged_dict:  # Only add if not already present
                merged_dict[key] = value
        
        hparams_obj = Namespace(**merged_dict)

        # Create a new model instance using the merged object
        model = BrepSeg(hparams_obj)

        model.load_state_dict(checkpoint_data['state_dict'])

        print("--- Checkpoint loaded successfully ---")

        # Create a more appropriate output directory structure
        import os
        # Use current directory or create a 'test_output' folder
        output_dir = os.path.join(os.getcwd(), "test_output", "features")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
        # Set environment variable that the model might use for output path
        os.environ['FEATURE_OUTPUT_DIR'] = output_dir

        test_data = Dataset(root_dir=args.dataset_path, split="test", random_rotate=False, num_class=args.num_classes)
        test_loader = test_data.get_dataloader(
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        
        trainer.test(model, dataloaders=test_loader)