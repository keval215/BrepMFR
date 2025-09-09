#!/usr/bin/env python3
"""
run_training_g6_2x.py - FIXED: Single ml.g6.48xlarge Training
- Uses 1x ml.g6.48xlarge instance (8x L4 GPUs)  
- Fixed hyperparameters to match actual argparse arguments
- Checkpointing enabled for Spot interruption recovery
- Robust error handling
"""
import os
import sagemaker
from sagemaker.pytorch import PyTorch
from botocore.exceptions import ClientError

# 1) Session and role
sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# 2) S3 paths (keep your existing bucket/prefix)
bucket = "brepmfr-training"
data_prefix = "data"
s3_data_root = f"s3://{bucket}/{data_prefix}"
s3_train_uri = f"{s3_data_root}/test"
s3_val_uri = f"{s3_data_root}/test"

print("🚀 LAUNCHING SINGLE ml.g6.48xlarge DISTRIBUTED TRAINING")
print("=" * 80)
print("Configuration:")
print("  Instance type: ml.g6.48xlarge (8x L4 GPUs)")
print("  Instance count: 1")
print("  Training data:", s3_train_uri)
print("  Validation data:", s3_val_uri)
print("=" * 80)

# 3) Estimator configuration
checkpoint_s3_uri = f"s3://{bucket}/checkpoints/"

estimator = PyTorch(
    entry_point="train_entry.py",
    source_dir=".",
    role=role,
    
    # Framework settings
    framework_version="1.13.1",
    py_version="py39",
    
    # Instance configuration - SINGLE NODE
    instance_type="ml.g6.48xlarge",  # L4-based GPU instance  
    instance_count=1,                # Single node with 8x L4 GPUs
    
    # Enable PyTorch distributed across GPUs
    distribution={"torch_distributed": {"enabled": True}},
    
    # Spot settings for cost optimization
    use_spot_instances=True,
    max_run=7200,     # 2 hours of actual training time
    max_wait=14400,   # up to 4 hours waiting for Spot capacity
    
    # Checkpointing for Spot interruption recovery
    checkpoint_s3_uri=checkpoint_s3_uri,
    checkpoint_local_path="/opt/ml/checkpoints",
    
    # FIXED: Only use hyperparameters that exist in your argparse
    hyperparameters={
        # Basic training parameters (these exist in your script)
        "traintest": "train",
        "dataset": "cadsynth",
        "dataset_path": "/opt/ml/input/data",
        "batch_size": 16,           # Increased for 8 GPUs
        "num_workers": 6,           # Good for single node
        "max_epochs": 100,
        "num_classes": 25,
        "nodes": 1,                # Single node
        "experiment_name": "BrepMFR_G6_Training",
        
        # Model architecture parameters (these exist in your script)
        "dropout": 0.3,
        "attention_dropout": 0.3,
        "d_model": 512,
        "dim_node": 256,
        "n_heads": 32,
        "n_layers_encode": 8,
        
        # REMOVED: These don't exist in your argparse and cause failures
        # "auto_resume": True,        
        # "checkpoint_dir": "/opt/ml/checkpoints",
        # "checkpoint_frequency": 5,  
        # "save_top_k": 3,           
        # "sync_batchnorm": True,     
        # "find_unused_parameters": False,  
        # "mixed_precision": True,    
        # "gradient_clip_val": 1.0,  
        # "learning_rate": 0.001,
        # "weight_decay": 0.0001,
    },
    
    output_path=f"s3://{bucket}/model-output-g6-single/",
    
    enable_sagemaker_metrics=True,
    debugger_hook_config=False,
    input_mode="File",
    
    # Multi-GPU environment variables
    environment={
        "PYTHONUNBUFFERED": "1",
        "SM_LOG_LEVEL": "10",      # More verbose logging
        
        # NCCL settings for multi-GPU communication
        "NCCL_DEBUG": "INFO",      # More verbose NCCL logging
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_TREE_THRESHOLD": "0",
        "NCCL_IB_DISABLE": "1",     # Disable InfiniBand for L4
        
        # PyTorch distributed settings
        "TORCH_DISTRIBUTED_DEBUG": "INFO",
        "CUDA_LAUNCH_BLOCKING": "0",
        "TORCH_CUDNN_V8_API_ENABLED": "1",
        
        # Ensure proper GPU visibility
        "NVIDIA_VISIBLE_DEVICES": "all",
        
        # Training debugging
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    },
)

def launch_training():
    """Launch training with robust error handling"""
    try:
        print("🚀 Submitting single-node multi-GPU training job...")
        print(f"   Instance type: ml.g6.48xlarge x1")
        print(f"   Total GPUs: 8x L4 (24GB each)")
        print(f"   Total VRAM: ~192GB")
        print(f"   Estimated Spot cost: ~$4-6/hour (vs ~$16/hour On-Demand)")
        print(f"   Max training time: {estimator.max_run}s ({estimator.max_run/3600:.1f}h)")
        print(f"   Max wait time: {estimator.max_wait}s ({estimator.max_wait/3600:.1f}h)")
        print(f"   Checkpoint location: {checkpoint_s3_uri}")
        print()
        
        estimator.fit(
            {"train": s3_train_uri, "validation": s3_val_uri}, 
            wait=True, 
            logs=True
        )
        
        print("🎉 Training completed successfully!")
        print("=" * 80)
        print("Job details:")
        print(f"  Job name: {estimator.latest_training_job.name}")
        print(f"  Model artifacts: {estimator.model_data}")
        print(f"  Checkpoint location: {checkpoint_s3_uri}")
        print("=" * 80)
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        print("❌ Training job failed with AWS error:")
        print(f"   Error Code: {error_code}")
        print(f"   Error Message: {error_message}")
        
        if error_code == 'ResourceLimitExceeded':
            print("\n💡 Resource not available. Suggestions:")
            print("   1. Try a different region (us-west-2, us-east-1)")
            print("   2. Try ml.g5.48xlarge instead of ml.g6.48xlarge")
            print("   3. Wait and retry during off-peak hours")
            print("   4. Request quota increase for ml.g6.48xlarge")
        
        if hasattr(estimator, "latest_training_job") and estimator.latest_training_job is not None:
            job_name = estimator.latest_training_job.name
            region = sess.boto_region_name
            print(f"\n   Job name: {job_name}")
            print(f"   CloudWatch logs: https://console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:log-groups/log-group/%2Faws%2Fsagemaker%2FTrainingJobs/log-events/{job_name}")
        else:
            print("   No training job was created")
            
        return False
        
    except Exception as e:
        print("❌ Training job failed with unexpected error:")
        print(f"   Error: {str(e)}")
        
        if hasattr(estimator, "latest_training_job") and estimator.latest_training_job is not None:
            job_name = estimator.latest_training_job.name
            region = sess.boto_region_name
            print(f"   Job name: {job_name}")
            print(f"   CloudWatch logs: https://console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:log-groups/log-group/%2Faws%2Fsagemaker%2FTrainingJobs/log-events/{job_name}")
        
        import traceback
        traceback.print_exc()
        return False

# 4. Launch training
if __name__ == "__main__":
    print("Starting ml.g6.48xlarge single-node distributed training...")
    
    success = launch_training()
    
    if success:
        print("\n✅ Training job submitted successfully!")
        print("Monitor progress in SageMaker console or CloudWatch logs")
    else:
        print("\n❌ Failed to start training job")
        
        # Provide fallback suggestions
        print("\n🔄 Fallback options:")
        print("1. Check CloudWatch logs for detailed error messages")
        print("2. Try ml.g5.48xlarge (8x A10G GPUs, better availability)")
        print("3. Switch to different AWS region")
        print("4. Use On-Demand instead of Spot instances")
        
        exit(1)
    
print("\n🎯 Script completed!")