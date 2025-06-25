# HitAnno
This repository contains the official implementation of the paper: 
**HitAnno: a hierarchical transformer for cell type annotation on scATAC-seq data.**

Table of Contents:
- [Overview](#Overview)
- [Installation](#Installation)
- [Usage](#Usage)
- [Tutorial](#Tutorial)
- [Citation](#Citation)
- [License](#License)

## Overview
We developed HitAnno, a scalable model built on a hierarchical transformer architecture for accurate cell type annotation on large scATAC-seq datasets. HitAnno constructs “cell sentences” by leveraging accessibility profiles on cell-type-specific peaks, capturing the epigenomic cell heterogeneity. The model adopts a two-level attention mechanism to capture both peak-level and peak-set-level dependencies, enabling hierarchical feature integration for reliable annotation performance.

<div align="center">
  <img src="figures/model.png" alt="HitAnno model" width="70%">
</div>

## Installation
We recommend creating a new virtual environment to ensure compatibility:
```bash
conda create -n hitanno python=3.9
conda activate hitanno
```
Install core computation and plotting libraries:
```bash
conda install numpy==1.26.4 pandas==2.2.2 scipy==1.13.0
conda install seaborn==0.13.2
```
Install PyTorch with CUDA support. The following command assumes a GPU compatible with CUDA 11.8:
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install libraries required for data I/O, analysis, and modeling:
```bash
pip install hdf5storage==0.1.19
pip install scanpy==1.10.1
pip install transformers==4.40.1
```
Install FlashAttention. Refer to the [FlashAttention installation guide](https://github.com/Dao-AILab/flash-attention#installation) for details:
```bash
pip install flash-attn==2.5.8 --no-build-isolation
```
(Optional) If you use Jupyter Notebooks, register the environment kernel:
```bash
pip install ipykernel
python -m ipykernel install --user --name=hitanno
```

## Usage
*under construction*

## Tutorial
*under construction*

## Citation
*Coming soon*

## License
This project is covered under the MIT license.
