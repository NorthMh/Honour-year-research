# NAME OF PROJ(to be created)
This is an official code of XXX.

## Overview
To be complete...

## Installation
### Create environments
Asssume you are in the `path/to/this/proj/`.


Stable Virtual Camera:
```bash
conda create -n svc python=3.11 -y
conda activate svc
git clone --recursive https://github.com/Stability-AI/stable-virtual-camera
cd stable-virtual-camera
pip install -e .
pip install einops
pip install safetensors
mkdir output
# you may face this problem: https://github.com/Stability-AI/stable-virtual-camera/issues/92
sed -i 's|stabilityai/stable-diffusion-2-1-base|Manojb/stable-diffusion-2-1-base|g' seva/modules/autoencoder.py
conda deactivate
```
DiffMorpher:
```bash
git clone https://github.com/Kevin-thu/DiffMorpher.git
cd DiffMorpher
conda create -n diffmorph python=3.11 -y
conda activate diffmorph
pip install -r requirements.txt
# you may need to install below
conda install -c conda-forge libgl -y
conda install -c conda-forge glib -y
#
conda deactivate
mkdir third_party
cd third_party
```
sam3:
```bash
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install -e ".[notebooks]"
pip install -e ".[train,dev]"
pip install openai
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless
pip install -U ultralytics
huggingface-cli download facebook/sam3 sam3.pt --local-dir .
conda deactivate
cd ..
```
GeoAware-SC:
```bash
conda create -n geo-aware python=3.9 -y
conda activate geo-aware
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev -y
git clone https://github.com/Junyi42/GeoAware-SC.git
cd GeoAware-SC
pip install pyproject-toml && pip install setuptools wheel && pip install -e . --no-build-isolation
pip install xformers==0.0.16
pip install pillow
# NOTE: you may need to replace two files in hub/facebookresearch_dinov2_main/dinov2/layers/attention.py & block.py with the one in replace/third_party/attention.py & block.py after first running.
# NOTE: you may also face cuFFT error: see https://github.com/Junyi42/GeoAware-SC/issues/1 for more details
conda deactivate
cd ..
```

### Replace & Add files
Please:
1. replace `main.py`, `model.py` and `utils/lora_utils.py` in the original DiffMorpher folder with the one in `replace/DiffMorpher`.
2. replace `demo.py` and `seva/geometry.py` in the original stable-virtual-camera folder with the one in `replace/stable-virtual-camera`.
3. add `get_seg.py` and `feat_matching.py` from `replace/third_party` to `DiffMorpher/third_party/sam3` and `DiffMorpher/third_party/GeoAware-SC` respectively
## Usage
To be complete...
## Dateset
To be complete...
## Q&A
To be complete...
## Citing
To be complete...
