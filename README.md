<div align="center">
<h1>MeTRIC: Metric Temporally-consistent Reconstruction In Causal Streaming</h1>
</div>

### [Paper]TODO  | [Project Page]TODO | [Online Demo]TODO

>Metric Temporally-consistent Reconstrution In Causal Streaming

>Porter Dosch<sup>\*</sup>, Luise Haller<sup>\*</sup>, Faisal Zaghloul, James Tompkin

<sup>*</sup> Equal contribution.

**MeTRIC**, a causal transformer architecture for **temporally consistent 4D geometry generation** built off of [StreamVGGT](https://github.com/wzzheng/StreamVGGT), delivers both fast inference and temporally-consistent 4D reconstruction.

## News
TODO: as project progresses, update this section

## Overview
TODO: add overview of method once everything finalized

### Installation

1. Clone MeTRIC
```bash
git clone https://github.com/jPorterDosch/MeTRIC.git
cd MeTRIC
```
2. Create conda environment
```bash
conda create -n MeTRIC python=3.11 cmake=3.14.0
conda activate MeTRIC 
```

3. Install requirements
```bash
pip install -r requirements.txt
conda install 'llvm-openmp<16'
```

### Download Checkpoints
Pre-trained StreamVGGT is also available at both [Hugging Face](https://huggingface.co/lch01/StreamVGGT/) and [Tsinghua cloud](https://cloud.tsinghua.edu.cn/d/d6ad8f36fcd541bcb246/).


## Data Preparation
### Training Datasets
TODO: instructions for downloading/preprocessing datasets

### Evaluation Datasets
Please refer to [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare Sintel, Bonn, KITTI, NYU-v2, ScanNet, 7scenes and Neural-RGBD datasets. Preprocessing scripts are available in `datasets_preprocess`.

## Folder Structure
The overall folder structure should be organized as follows：
TODO: fill in rest of this once we have decided on directory structure
```
MeTRIC
├── ckpt/
|   ├── model.pt
|   └── checkpoints.pth
```

## Evaluation
TODO: add eval scripts


## Demo
We provide a demo for StreamVGGT, based on the demo code from [VGGT](https://github.com/facebookresearch/vggt). You can follow the instructions below to launch it locally or try it out directly on [Hugging Face](https://huggingface.co/spaces/lch01/StreamVGGT).
```bash
pip install -r requirements_demo.txt
python demo_gradio.py
```

**Note**: While StreamVGGT typically reconstructs a scene in under one second, 3D point visualization may take much longer due to slower third-party rendering.

## Acknowledgements
Our code is based on the following repositories:

[StreamVGGT](https://github.com/wzzheng/streamvggt)
[DUSt3R](https://github.com/naver/dust3r)
[MonST3R](https://github.com/Junyi42/monst3r.git)
[Spann3R](https://github.com/HengyiWang/spann3r.git)
[CUT3R](https://github.com/CUT3R/CUT3R)
[VGGT](https://github.com/facebookresearch/vggt)
[Point3R](https://github.com/YkiWu/Point3R)

## Citation
TODO: add citation after any paper is written/uploaded
