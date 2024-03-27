6D Rotation Representation for Unconstrained Head Pose Estimation

### Note

* The default train dataset is `300W-LP`
* The default test dataset is `AFLW2000`

### Installation

```
conda create -n PyTorch python=3.11
conda activate PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python==4.5.5.64
pip install scipy
pip install tqdm
pip install timm
```

### Train

* Configure your dataset path in `main.py` for training
* Download [IMAGENET](https://github.com/jahongir7174/SixDRepNet/releases/tag/v0.0.1) pretrained weights
* Run `python main.py --train` for Single-GPU training
* Run `bash main.sh $ --train` for Multi-GPU training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Demo

* Configure your video path in `main.py` for visualizing the demo
* Run `python main.py --demo` for demo

### Results

|   Backbone   | Epochs | Pitch |  Yaw | Roll |  MAE | Parameters(M) | FLOPS (B) | Throughput (images/s) |
|:------------:|:------:|------:|-----:|-----:|-----:|--------------:|----------:|----------------------:|
|  RepNet-A2   |   90   |  4.78 | 3.68 | 3.25 | 3.90 |         25.49 |       5.1 |                  1322 |
| RepNet-B1G2* |   30   |  4.91 | 3.63 | 3.37 | 3.97 |         41.36 |       8.8 |                   792 |

`*` means that the results are from original repo, see reference

![Alt Text](./demo/demo.gif)

#### Reference

* https://github.com/thohemp/6DRepNet
