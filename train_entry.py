#!/usr/bin/env python3
"""
train_entry.py - FINAL VERSION
✅ Dependency installation only on RANK=0
✅ No pip upgrade (avoid corruption)
✅ Clean URLs, robust checkpointing, error recording
"""

import subprocess
import sys
import os
import logging
import time
import importlib
from pathlib import Path
from torch.distributed.elastic.multiprocessing.errors import record

# Only import torch after installation (avoid premature import)
def setup_logging():
    """Setup logging for better debugging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/opt/ml/output/training.log') if os.path.exists('/opt/ml/output') else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


class RobustCheckpointManager:
    """Handles checkpointing with Spot instance interruption recovery"""
    
    def __init__(self, checkpoint_dir, rank=0, world_size=1):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        
        logging.info(f"Checkpoint manager initialized: rank={rank}, world_size={world_size}")
        logging.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def find_latest_checkpoint(self):
        """Find the most recent checkpoint file"""
        checkpoint_patterns = [
            "last.ckpt",
            "epoch=*.ckpt", 
            "checkpoint-*.ckpt",
            "*.ckpt"
        ]
        
        logging.info(f"Searching for checkpoints in: {self.checkpoint_dir}")
        
        for pattern in checkpoint_patterns:
            checkpoints = list(self.checkpoint_dir.glob(pattern))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                logging.info(f"Found checkpoint: {latest_checkpoint}")
                
                if self.verify_checkpoint(latest_checkpoint):
                    return str(latest_checkpoint)
                else:
                    logging.warning(f"Checkpoint {latest_checkpoint} is corrupted, skipping...")
        
        logging.info("No valid checkpoint found, starting fresh training")
        return None
    
    def verify_checkpoint(self, checkpoint_path):
        """Verify checkpoint file is valid and loadable"""
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            required_keys = ['epoch', 'model_state_dict']
            return all(key in checkpoint for key in required_keys)
        except Exception as e:
            logging.warning(f"Checkpoint verification failed: {e}")
            return False
    
    def should_save_checkpoint(self, epoch, save_frequency=5):
        return self.is_main_process and (epoch % save_frequency == 0 or epoch == 0)
    
    def save_checkpoint(self, model, optimizer, lr_scheduler, epoch, metrics=None):
        if not self.is_main_process:
            return
            
        try:
            import torch
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                'metrics': metrics or {},
                'world_size': self.world_size,
                'checkpoint_version': '1.0'
            }
            
            # Save main and epoch-specific checkpoints
            (self.checkpoint_dir / "last.ckpt").write_bytes(
                torch.save(checkpoint_data, None)
            )
            (self.checkpoint_dir / f"epoch={epoch:03d}.ckpt").write_bytes(
                torch.save(checkpoint_data, None)
            )
            
            logging.info(f"Checkpoint saved at epoch {epoch}")
            self.cleanup_old_checkpoints(keep_last=3)
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, model, optimizer=None, lr_scheduler=None):
        checkpoint_path = self.find_latest_checkpoint()
        if not checkpoint_path:
            return 0, {}
        
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            metrics = checkpoint.get('metrics', {})
            
            logging.info(f"Resuming from epoch {start_epoch}")
            return start_epoch, metrics
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return 0, {}

    def cleanup_old_checkpoints(self, keep_last=3):
        if not self.is_main_process:
            return
        try:
            epoch_checkpoints = sorted(
                self.checkpoint_dir.glob("epoch=*.ckpt"),
                key=lambda x: int(x.stem.split('=')[1])
            )
            for ckpt in epoch_checkpoints[:-keep_last]:
                ckpt.unlink(missing_ok=True)
                logging.info(f"Removed old checkpoint: {ckpt}")
        except Exception as e:
            logging.warning(f"Cleanup failed: {e}")


def verify_pip_integrity():
    """Check if pip is available"""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"],
                                capture_output=True, text=True, check=True)
        logging.info(f"pip available: {result.stdout.strip()}")
        return True
    except Exception as e:
        logging.error(f"pip not available: {e}")
        return False


def safe_pip_install(packages, extra_args=None, allow_failure=False):
    """Safely install packages with retries"""
    logger = logging.getLogger(__name__)
    
    if not verify_pip_integrity():
        logger.error("pip not available")
        return False

    cmd = [sys.executable, "-m", "pip", "install"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)

    logger.info(f"Running: {' '.join(cmd)}")
    
    for attempt in range(3):
        try:
            subprocess.run(cmd, check=True, capture_output=False)
            logger.info(f"✅ Installed: {packages}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Install failed (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            elif not allow_failure:
                return False
    
    if allow_failure:
        logger.warning(f"Failed to install {packages} after 3 attempts, but continuing as allowed.")
        return True
    return False


@record  # Enables detailed error reporting from torchelastic
def install_all_dependencies():
    """Install dependencies only on RANK=0"""
    logger = logging.getLogger(__name__)
    rank = int(os.environ.get("RANK", "0"))

    # 🔐 Only rank 0 installs
    if rank != 0:
        logger.info(f"Rank {rank}: Skipping dependency install (only rank 0 runs it)")
        return True

    logger.info("=" * 80)
    logger.info("🔧 INSTALLING DEPENDENCIES (RANK 0 ONLY)")
    logger.info("=" * 80)

    if not verify_pip_integrity():
        logger.error("❌ pip is not functional")
        return False

    # ❌ DO NOT UPGRADE PIP — it breaks in distributed environments
    logger.info("✅ Skipping pip upgrade for safety")

    # NEW: Downgrade pip to a version compatible with pytorch-lightning==1.7.1 metadata
    logger.info("🔧 Downgrading pip to version 23.3.1 for compatibility with pytorch-lightning==1.7.1...")
    if not safe_pip_install(["pip==23.3.1"], ["--force-reinstall", "--no-deps"]):
        logger.error("❌ Failed to downgrade pip to 23.3.1")
        return False
    logger.info("✅ pip downgraded to 23.3.1")

    # Verify new pip version
    if not verify_pip_integrity():
        logger.error("❌ pip is not functional after downgrade")
        return False

    # Fix URLs: no extra spaces
    pytorch_url = "https://download.pytorch.org/whl/cu116"
    pyg_url = "https://data.pyg.org/whl/torch-1.13.0+cu116.html"

    
        # Step 1: PyTorch (must pin torch/vision/audio to same CUDA version!)
    pytorch_url = "https://download.pytorch.org/whl/cu117"
    if not safe_pip_install([
        "torch==1.13.1+cu117",
        "torchvision==0.14.1+cu117",
        "torchaudio==0.13.1+cu117"
    ], ["--index-url", pytorch_url]):
        return False


    # Step 2: Core packages
    if not safe_pip_install(["numpy==1.23.5"]):
        return False
    if not safe_pip_install(["torchmetrics==0.11.4"]): # Install torchmetrics separately
        return False

    # Step 3: PyTorch Lightning - Install dependencies first, then Lightning itself with --no-deps
    logger.info("🔧 Installing PyTorch Lightning 1.7.1 dependencies...")
    lightning_deps = [
        "pyDeprecate>=0.3.1,<0.4.0",
        "tensorboard>=2.9.1",
        "lightning-utilities>=0.3.0,<0.9.0"
    ]
    for dep in lightning_deps:
        if not safe_pip_install([dep], allow_failure=True): # Allow some deps to fail
            logger.warning(f"⚠️ Could not install {dep}, continuing...")

    logger.info("🔧 Installing PyTorch Lightning 1.7.1 itself (--no-deps)...")
    # Now install Lightning itself without its dependencies
    if not safe_pip_install(["pytorch-lightning==1.7.1"], ["--no-deps"]):
        logger.error("❌ Failed to install pytorch-lightning==1.7.1 even with pip 23.3.1")
        return False
    logger.info("✅ PyTorch Lightning 1.7.1 installed")

    # Step 4: DGL & Geometric
    if not safe_pip_install(["dgl==1.0.0"]):
        return False
    if not safe_pip_install(["torch-geometric==2.2.0"]):
        return False

    # Step 5: Geometric extensions
    geo_pkgs = [
        "torch-scatter==2.1.1+pt113cu116",
        "torch-sparse==0.6.17+pt113cu116",
        "torch-cluster==1.6.1+pt113cu116",
        "torch-spline-conv==1.2.2+pt113cu116"
    ]
    for pkg in geo_pkgs:
        if not safe_pip_install([pkg], ["-f", pyg_url], allow_failure=True):
            base = pkg.split('+')[0]
            logger.warning(f"⚠️ Falling back to CPU version: {base}")
            if not safe_pip_install([base], allow_failure=True):
                logger.warning(f"⚠️ Could not install {base} either, continuing...")

    # Step 6: Critical missing deps
    if not safe_pip_install(["prefetch_generator"]):
        return False
    if not safe_pip_install(["fairseq"]):
        return False

    # Step 7: Additional
    extras = [
        "setuptools>=60.0.0,<70.0.0", "wheel>=0.37.0",
        "typing-extensions>=4.0.0", "fsspec>=2021.05.0",
        "packaging>=17.0", "PyYAML>=5.4", "tqdm>=4.57.0"
    ]
    for pkg in extras:
        safe_pip_install([pkg], allow_failure=True) # Allow these to fail if already present

    logger.info("🎉 Dependencies installed successfully on rank 0")
    return True


def verify_installations():
    logger = logging.getLogger(__name__)
    logger.info("🔍 Starting package installation verification...\n")

    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("torchaudio", "TorchAudio"),
        ("numpy", "NumPy"),
        ("dgl", "Deep Graph Library"),
        ("torch_geometric", "PyTorch Geometric"),
        ("torch_scatter", "Torch Scatter"),
        ("torch_sparse", "Torch Sparse"),
        ("torch_cluster", "Torch Cluster"),
        ("torch_spline_conv", "Torch Spline Conv"),
        ("torchmetrics", "TorchMetrics"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("prefetch_generator", "Prefetch Generator"),
        ("fairseq", "FairSeq"),
    ]

    results = {}
    failed = []

    for module_name, display_name in packages:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            logger.info(f"✅ {display_name} ({module_name}) v{version}")
            results[module_name] = {"status": "ok", "version": version}
        except ImportError as e:
            logger.error(f"❌ {display_name} ({module_name}) not found: {e}")
            failed.append(module_name)
            results[module_name] = {"status": "missing"}

    # Critical packages that must exist
    critical = ["torch", "pytorch_lightning", "prefetch_generator", "fairseq"]
    critical_failures = [f for f in failed if f in critical]

    if critical_failures:
        logger.error(f"\n❌ Critical package verification failed: {', '.join(critical_failures)}")
        return {"ok": False, "results": results}

    logger.info("\n✅ Package verification completed successfully.")
    return {"ok": True, "results": results}


def check_data_paths():
    logger = logging.getLogger(__name__)
    data_path = "/opt/ml/input/data"
    if not os.path.exists(data_path):
        logger.error(f"❌ Data path not found: {data_path}")
        return False
    train_path = os.path.join(data_path, "train")
    if not os.path.exists(train_path):
        logger.error(f"❌ Training data not found: {train_path}")
        return False
    logger.info(f"✅ Training data found: {train_path}")
    return True


def setup_checkpoint_manager():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    checkpoint_dir = os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints')
    return RobustCheckpointManager(checkpoint_dir, rank, world_size)


def prepare_training_environment():
    logger = logging.getLogger(__name__)
    env_vars = {
        'PYTHONUNBUFFERED': '1',
        'NCCL_DEBUG': 'WARN',
        'NCCL_SOCKET_IFNAME': 'eth0'
    }
    for k, v in env_vars.items():
        if k not in os.environ:
            os.environ[k] = v
            logger.info(f"Set: {k}={v}")

    for d in ['/opt/ml/checkpoints', '/opt/ml/output', '/opt/ml/model']:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Dir: {d}")


@record  # 🔥 Enables detailed error reporting
def main():
    logger = setup_logging()
    logger.info("🚀 SAGEMAKER TRAINING JOB - RANK-0 INSTALL + ROBUST CHECKPOINTING")
    logger.info("=" * 80)

    try:
        prepare_training_environment()

        # 🔐 Install only on rank 0
        if not install_all_dependencies():
            logger.error("❌ Dependency installation failed")
            sys.exit(1)

        # Wait for all processes after install
        import torch  # ✅ Add the missing import here
        if torch.distributed.is_available():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            torch.distributed.barrier()

        if not verify_installations():
            logger.error("❌ Verification failed")
            sys.exit(1)

        if not check_data_paths():
            logger.error("❌ Data not found")
            sys.exit(1)

        checkpoint_manager = setup_checkpoint_manager()
        logger.info("✅ Checkpoint manager ready")

        logger.info("📥 Importing segmentation module...")
        import segmentation
        logger.info("✅ Module imported")

        if hasattr(segmentation, 'main'):
            try:
                result = segmentation.main(checkpoint_manager=checkpoint_manager)
            except TypeError:
                result = segmentation.main()
            logger.info(f"🎉 Training completed: {result}")
        else:
            import builtins
            builtins.checkpoint_manager = checkpoint_manager
            logger.info("🎉 Script executed")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()