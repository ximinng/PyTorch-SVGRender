Installation
===============

.. _install:

You can follow the steps below to quickly get up and running with PyTorch-SVGRender.
These steps will let you run quick inference locally.

In the top level directory run,

.. code-block:: console

   $ sh script/install.sh

Note: Make sure that the script file has execution **permissions** (you can give them using ``chmod +x script.sh``), and
then run the script.

**If you want to install it yourself step by step, you can refer to the following content,**

Create a new conda environment:

.. code-block:: console

   (base) $ conda create --name svgrender python=3.10
   (base) $ conda activate svgrender

Install pytorch and the following libraries:

.. code-block:: console

   (svgrender) $ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
   (svgrender) $ pip install hydra-core omegaconf
   (svgrender) $ pip install freetype-py shapely svgutils
   (svgrender) $ pip install opencv-python scikit-image matplotlib visdom wandb BeautifulSoup4
   (svgrender) $ pip install triton numba
   (svgrender) $ pip install numpy scipy scikit-fmm einops timm fairscale==0.4.13
   (svgrender) $ pip install accelerate transformers safetensors datasets

Install LaMa:

.. code-block:: console

   (svgrender) $ pip install easydict scikit-learn pytorch_lightning webdataset
   (svgrender) $ pip install albumentations==0.5.2
   (svgrender) $ pip install kornia==0.5.0
   (svgrender) $ pip install wldhx.yadisk-direct

   (svgrender) $ cd lama
   (svgrender) $ curl -O -L https://huggingface.co/xingxm/PyTorch-SVGRender-models/resolve/main/big-lama.zip
   (svgrender) $ unzip big-lama.zip
   (svgrender) $ cd ..

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