# MIST: Multi-modal Iterative Spatial-Temporal Transformer for Long-form Video Question Answering
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.11.0-%237732a8)

[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_MIST_Multi-Modal_Iterative_Spatial-Temporal_Transformer_for_Long-Form_Video_Question_Answering_CVPR_2023_paper.pdf)]

> **MIST: Multi-modal Iterative Spatial-Temporal Transformer for Long-form Video Question Answering**
> <br>Difei Gao, Luowei Zhou, Lei Ji, Linchao Zhu, Yi Yang, Mike Zheng Shou<br>



### Requirements

1. Pytorch > 1.9.0 (Tested on Pytorch 1.11.0 with CUDA 11.3)
2. We have performed experiments on NVIDIA GeForce A5000 GPU 24GB GPU
3. See [`requirements.txt`](requirements.txt) for the required python packages and run to install them

Let's begin from creating and activating a Conda environment an virtual environment 
```
conda create --name mistenv python=3.7
conda activate mistenv
```
Then, clone this repository and install the requirements.
```
$ git clone https://github.com/showlab/mist.git
$ cd mist
$ pip install -r requirements.txt
```

### Data Preparation
You can either download the feature from our shared drive or extract by your own with the given script.

1. Option 1: Download features from online drive:

[Open Google Drive Folder]()

This folder provides the dataset annotation files and features.

2. Option 2: Extract by running the script in [`extract\extract_clip_features.ipynb`].
The code extract the patch features of the video frames.

### Training
Simply run the shell [`agqa_v2_mist.sh`] in the [`shells\`] to start training.
```
./shells/agqa_v2_mist.sh
```
or input the command below on the terminal.
```


```


### Ackonwledgements
We are grateful to just-ask, an excellent VQA codebase, on which our codes are developed.

### Bibtex
```
@inproceedings{gao2023mist,
  title={MIST: Multi-modal Iterative Spatial-Temporal Transformer for Long-form Video Question Answering}, 
  author={Difei Gao and Luowei Zhou and Lei Ji and Linchao Zhu and Yi Yang and Mike Zheng Shou},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14773--14783},
  year={2023}
}
```
