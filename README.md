# BrepMFR

Code for BrepMFR: Enhancing machining feature recognition in B-rep models through deep learning and domain adaptation.

![The network architecture of BrepMFR](docs/img/network_architecture.jpg)

## About BrepMFR

BrepMFR, a novel deep learning network designed for machining feature recognition on B-rep models within the CAD/CAM domain. The original B-rep model is converted into a graph representation for network-friendly input, where graph nodes and edges respectively correspond to faces and edges of the original model. Subsequently, we leverage a graph neural network based on the Transformer architecture and graph attention mechanism to encode both local geometric shape and global topological relationships, achieving high-level semantic extraction and prediction of machining feature categories. Furthermore, to enhance the performance of neural networks on real-world CAD models, we adopt a two-step training strategy within a novel transfer learning framework.

## Preparation

### Environment setup

```
git clone https://github.com/zhangshuming0668/BrepMFR.git
cd BrepMFR
conda env create -f environment.yml
conda activate brep_mfr
```

### Data preparation

Our synthetic CAD datasets have been publicly available on [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=931c088fd44f4d3e82891a5180f10d90)

## Training

For machining feature recognition, the network can be trained using:
```
python segmentation.py train --dataset_path /path/to/dataset --max_epochs 1000 --batch_size 64
```

The logs and checkpoints will be stored in a folder called `results/BrepMFR` based on the experiment name and timestamp, and can be monitored with Tensorboard:

```
tensorboard --logdir results/<experiment_name>
```

## Testing

The best checkpoints based on the smallest validation loss are saved in the results folder. The checkpoints can be used to test the model as follows:

```
python segmentation.py test --dataset_path /path/to/dataset --checkpoint ./results/BrepMFR/best.ckpt --batch_size 64
```



AlgorithmError: ExecuteUserScriptError: ExitCode 1 ErrorMessage "TypeError: can't multiply sequence by non-int of type 'float' During handling of the above exception, another exception occurred Traceback (most recent call last) File "/opt/ml/code/train_entry.py", line 439, in main result = segmentation.main() File "/opt/ml/code/segmentation.py", line 187, in main model = BrepSeg(args) File "/opt/ml/code/models/brepseg_model.py", line 91, in __init__ self.brep_encoder = BrepEncoder( File "/opt/ml/code/models/modules/brep_encoder.py", line 67, in __init__ self.graph_node_feature = GraphNodeFeature( File "/opt/ml/code/models/modules/layers/brep_encoder_layer.py", line 189, in __init__ in_channels=7, output_dims=int(0.5*hidden_dim) File "/opt/ml/code/train_entry.py", line 437, in main result = segmentation.main(checkpoint_manager=checkpoint_manager) ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 192) of binary: /opt/conda/bin/python3.9 File "/opt/c