Welcome to PyTorch-SVGRender documentation!
===================================

**Pytorch-SVGRender** is the go-to library for state-of-the-art differentiable rendering methods for image vectorization.

.. note::

   This project is under active development.

Table of Contents
----------

.. toctree::
   :maxdepth: 2

   live
   clipasso
   clipdraw
   styleclipdraw
   clipfont
   vectorfusion
   diffsketcher
   wordasimage
   svgdreamer
   api

Installation
----------

Create a new conda environment:

.. code-block:: console

   (base) $ conda create --name svgrender python=3.10
   (base) $ conda activate svgrender

Install pytorch and the following libraries:

.. code-block:: console

   (svgrender) $ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
   (svgrender) $ pip install hydra-core omegaconf
   (svgrender) $ pip install hydra-core omegaconfpip install freetype-py shapely svgutils
   (svgrender) $ pip install opencv-python scikit-image matplotlib visdom wandb BeautifulSoup4
   (svgrender) $ pip install triton numba
   (svgrender) $ pip install numpy scipy timm scikit-fmm einops
   (svgrender) $ pip install accelerate transformers safetensors datasets

Install LaMa:

.. code-block:: console

   (svgrender) $ pip install easydict scikit-learn pytorch_lightning webdataset
   (svgrender) $ pip install albumentations==0.5.2
   (svgrender) $ pip install kornia==0.5.0
   (svgrender) $ pip install wldhx.yadisk-direct

   (svgrender) $ cd lama
   (svgrender) $ curl -L $(yadisk-direct https://disk.yandex.ru/d/kHJkc7bs7mKIVA) -o big-lama.zip
   (svgrender) $ unzip big-lama.zip

Install CLIP:

.. code-block:: console

   (svgrender) $ pip install ftfy regex tqdm
   (svgrender) $ pip install git+https://github.com/openai/CLIP.git

Install diffusers:

.. code-block:: console

   (svgrender) $ pip install diffusers==0.20.2

Install xformers (require ``python=3.10``):

.. code-block:: console

   (svgrender) $ conda install xformers -c xformers

Install diffvg:

.. code-block:: console

   (svgrender) $ git clone https://github.com/BachiLi/diffvg.git
   (svgrender) $ cd diffvg
   (svgrender) $ git submodule update --init --recursive
   (svgrender) $ conda install -y -c anaconda cmake
   (svgrender) $ conda install -y -c conda-forge ffmpeg
   (svgrender) $ pip install svgwrite svgpathtools cssutils torch-tools
   (svgrender) $ python setup.py install
