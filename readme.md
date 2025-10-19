![Python >=3.11.11](https://img.shields.io/badge/Python->=3.11.11-yellow.svg)
![PyTorch >=2.2.2](https://img.shields.io/badge/PyTorch->=2.2.2-blue.svg)

# ReIDMamba: Learning Discriminative Features with Visual State Space Model for Person Re-Identification [[pdf]](wating)
The *official* repository for [ReIDMamba: Learning Discriminative Features with Visual State Space Model for Person Re-Identification](wating)[Accept to TMM!].

![](https://github.com/GuHY777/ReIDMamba/blob/master/figs/reidmamba.jpg)

## Requirements

### Installation
```bash
conda env create -f environment.yml
conda activate your_environment_name
```
We recommend to use one 24G RTX 4090 for training and evaluation. If you find some packages are missing, please install them manually. 


### Prepare Datasets

All the datasets should be downloaded on your own. All the datasets should be organized as follows:
```
dataroot
├── Market1501
│   └── bounding_box_train
│   └── bounding_box_test
│   └── query
├── MSMT17
│   └── bounding_box_train
│   └── bounding_box_test
│   └── query
├── ...
```

Please modify 'args.dataroot' to the corresponding path.

## Pre-trained Models 
|      Model      | Download |
|:---------------:| :------: |
|    MambaR-Small | [link](https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_small_patch16_224.pth) |

Please modify the pre-trained model path of '_backbones' in the 'model/reidmamba.py'.
Change '/root/data/.cache/models/mambar_small_patch16_224.pth' to your own pre-trained model path.

## Acceleration Candy for Evaluation
To accelerate the evaluation process, enter the 'evaluation/rank_cylib' directory. You need to install CMake, then run 'make all' in the bash. After that, set 'args.use_cython' to 'True' to benefit from the evaluation acceleration. For more details, please refer to the fast-reid repository.

```bash
# Navigate to the rank_cylib directory
cd evaluation/rank_cylib

# Run make all
make all
```


## Examples
```bash
# Training 256x128 model on Market1501 dataset
python main.py --gpus 0 --exp 'market1501_256' --dataset 'Market1501' --img_size '256,128'

# Training 384x128 model on Market1501 dataset
python main.py --gpus 0 --exp 'market1501_384' --dataset 'Market1501' --img_size '384,128'

# Training 384x128 model on Market1501 dataset with small slide window
python main.py --gpus 0 --exp 'market1501_384' --dataset 'Market1501' --img_size '384,128' --model_kwargs 'backbone_name(str)=mambar_small_patch16_224|drop_path_rate(float)=0.3|num_cls_tokens(int)=12|cls_reduce(int)=4|use_cid(bool)=1|stride_size(int)=12|num_branches(int)=3|token_fusion_type(str)=max'
```


## Citation

If you find this code useful for your research, please cite our paper

```
wating for ...
```

