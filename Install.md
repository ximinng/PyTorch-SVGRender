## Installation

Create a new conda environment:

```shell
conda create --name svgrender python=3.10
conda activate svgrender
```

Install pytorch and the following libraries:

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install hydra-core omegaconf 
pip install freetype-py shapely svgutils
pip install opencv-python scikit-image matplotlib visdom wandb BeautifulSoup4
pip install triton numba
pip install numpy scipy scikit-fmm einops timm fairscale==0.4.13
pip install accelerate transformers safetensors datasets
```

Install LaMa:

```shell
pip install easydict scikit-learn pytorch_lightning==2.1.0 webdataset
pip install albumentations==0.5.2
pip install kornia==0.5.0
pip install wldhx.yadisk-direct

cd lama
# download LaMa model weights
# raw link(deprecated): curl -L $(yadisk-direct https://disk.yandex.ru/d/kHJkc7bs7mKIVA) -o big-lama.zip
curl -O -L https://huggingface.co/xingxm/PyTorch-SVGRender-models/resolve/main/big-lama.zip
unzip big-lama.zip
```

Install CLIP:

```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Install diffusers:

```shell
pip install diffusers==0.20.2
```

Install xformers (require `python=3.10`):

```shell
conda install xformers -c xformers
```

Install diffvg:

```shell
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils torch-tools
python setup.py install
```