#!/bin/bash
eval "$(conda shell.bash hook)"

conda create --name svgrender python=3.10
conda activate svgrender

echo "The conda environment was successfully created"

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

echo "Pytorch installation is complete. version: 1.12.1"

pip install hydra-core omegaconf
pip install freetype-py shapely svgutils
pip install opencv-python scikit-image matplotlib visdom wandb BeautifulSoup4
pip install triton numba
pip install numpy scipy scikit-fmm einops timm fairscale=0.4.13
pip install accelerate transformers safetensors datasets

echo "The basic dependency library is installed."

pip install easydict scikit-learn pytorch_lightning webdataset
pip install albumentations==0.5.2
pip install kornia==0.5.0

# Noting: you can download the lama model when you need it,
# download LaMa model weights:

cd lama
curl -O -L https://huggingface.co/xingxm/PyTorch-SVGRender-models/resolve/main/big-lama.zip
unzip big-lama.zip

echo "LaMa installation is complete."

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

echo "CLIP installation is complete."

pip install diffusers==0.20.2

echo "Diffusers installation is complete. version: 0.20.2"

conda install xformers -c xformers

echo "xformers installation is complete."

git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils torch-tools
python setup.py install

echo "DiffVG installation is complete."

echo "the running environment has been successfully installed!!!"