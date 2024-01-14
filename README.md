<h1 id="ptsvg" align="center">Pytorch-SVGRender</h1>

<p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10-or?logo=python" alt="pyhton"></a>
    <a href="http://mozilla.org/MPL/2.0/"><img src="https://img.shields.io/badge/license-MPL2.0-orange" alt="license"></a>
    <a href="https://ximinng.github.io/PyTorch-SVGRender-project/"><img src="https://img.shields.io/badge/website-Gitpage-yellow" alt="website"></a>
    <a href="https://pytorch-svgrender.readthedocs.io/en/latest/index.html"><img src="https://img.shields.io/badge/docs-readthedocs-purple" alt="docs"></a>
</p>

<div align="center">
<img src="./assets/logo.png" style="width: 350px; height: 300px;" alt="Pytorch-SVGRender">
<p><strong>Pytorch-SVGRender: </strong>The go-to library for differentiable rendering methods for SVG generation.</p>
</div>
<p align="center">
    <a href="#recent-updates">Updates</a> •
    <a href="#table-of-contents">Table of Contents</a> •
    <a href="#installation">Installation</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#faq">FAQ</a> •
    <a href="#todo">TODO</a> •
    <a href="#acknowledgement">Acknowledgment</a> •
    <a href="#citation">Citation</a> •
    <a href="#licence">Licence</a>
</p>

<h2 align="center">Recent Updates</h2>

- [12/2023] 🔥 We open-sourced Pytorch-SVGRender V1.0.

<h2 align="center">Table of Contents</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

### 1. Image Vectorization

- DiffVG: Differentiable Vector Graphics Rasterization for Editing and Learning (`SIGGRAPH 2020`)

  [[Project]](https://people.csail.mit.edu/tzumao/diffvg/) [[Paper]](https://cseweb.ucsd.edu/~tzli/diffvg/diffvg.pdf) [[Code]](https://github.com/BachiLi/diffvg)

  DiffVG is a differentiable rasterizer for 2D vector graphics. **This repository is based on DiffVG.**

- LIVE: Towards Layer-wise Image Vectorization (`CVPR 2022`)

  [[Project]](https://ma-xu.github.io/LIVE/) [[Paper]](https://ma-xu.github.io/LIVE/index_files/CVPR22_LIVE_main.pdf) [[Code]](https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization)

- CLIPasso: Semantically-Aware Object Sketching (`SIGGRAPH 2022`)

  [[Project]](https://clipasso.github.io/clipasso/) [[Paper]](https://arxiv.org/abs/2202.05822) [[Code]](https://github.com/yael-vinker/CLIPasso)

- CLIPascene: Scene Sketching with Different Types and Levels of Abstraction (`ICCV 2023`)

  [[Project]](https://clipascene.github.io/CLIPascene/) [[Paper]](https://arxiv.org/abs/2211.17256) [[Code]](https://github.com/yael-vinker/SceneSketch)

### 2. Text-to-SVG Synthesis

- CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders (`NIPS 2022`)

  [[Paper]](https://arxiv.org/abs/2106.14843) [[Code]](https://github.com/kvfrans/clipdraw)

- StyleCLIPDraw: Coupling Content and Style in Text-to-Drawing Synthesis

  [[Live]](https://slideslive.com/38970834/styleclipdraw-coupling-content-and-style-in-texttodrawing-synthesis?ref=account-folder-92044-folders) [[Paper]](https://arxiv.org/abs/2202.12362) [[Code]](https://github.com/pschaldenbrand/StyleCLIPDraw)

- CLIPFont: Texture Guided Vector WordArt Generation (`BMVC 2022`)

  [[Paper]](https://bmvc2022.mpi-inf.mpg.de/0543.pdf) [[Code]](https://github.com/songyiren98/CLIPFont)

- VectorFusion: Text-to-SVG by Abstracting Pixel-Based Diffusion Models (`CVPR 2023`)

  [[Project]](https://vectorfusion.github.io/) [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Jain_VectorFusion_Text-to-SVG_by_Abstracting_Pixel-Based_Diffusion_Models_CVPR_2023_paper.pdf)

- DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models (`NIPS 2023`)

  [[Project]](https://ximinng.github.io/DiffSketcher-project/) [[Live]](https://neurips.cc/virtual/2023/poster/72425) [[Paper]](https://arxiv.org/abs/2306.14685) [[Code]](https://github.com/ximinng/DiffSketcher)

- Word-As-Image for Semantic Typography (`SIGGRAPH 2023`)

  [[Project]](https://wordasimage.github.io/Word-As-Image-Page/) [[Paper]](https://arxiv.org/abs/2303.01818) [[Code]](https://github.com/Shiriluz/Word-As-Image)

- SVGDreamer: Text Guided SVG Generation with Diffusion Model

  [[Project]](https://ximinng.github.io/SVGDreamer-project/) [[Paper]](https://arxiv.org/abs/2312.16476) [[code]](https://github.com/ximinng/SVGDreamer)

<h2 align="center">Installation</h2>

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
pip install numpy scipy scikit-fmm einops timm fairscale=0.4.13
pip install accelerate transformers safetensors datasets
```

Install LaMa:

```shell
pip install easydict scikit-learn pytorch_lightning webdataset 
pip install albumentations==0.5.2
pip install kornia==0.5.0
pip install wldhx.yadisk-direct

cd lama
# download LaMa model weights
curl -L $(yadisk-direct https://disk.yandex.ru/d/kHJkc7bs7mKIVA) -o big-lama.zip
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

<h2 align="center">Quickstart</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

**SVGDreamer** generates various styles of SVG based on text prompts. It supports the use of six vector primitives, including Iconography, Sketch, Pixel Art, Low-Poly, Painting, and Ink and Wash.

```shell
# iconography style:
## sydney opera house
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='Sydney opera house. oil painting. by Van Gogh' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.num_paths=512 result_path='./svgdreamer/Sydney' mv=True multirun=True
## crane
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='black and red ink. a crane in chinese style. ink art by MSchiffer. whimsical. rough sketch.' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/crane' mv=True multirun=True
## German shepherd
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='A colorful German shepherd in vector art. tending on artstation.' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/GermanShepherd' mv=True multirun=True
## Hogwarts
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='Hermione Granger. Ron. Harry Potter. Hogwarts.' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/Hogwarts' mv=True multirun=True
## ship on seas
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='Seascape. Ship on the high seas. Storm. High waves. Colored ink by Mikhail Garmash. Louis Jover. Victor Cheleg' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/ship' mv=True multirun=True

# low-ploy style:
## bald eagle
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='A picture of a bald eagle. low-ploy. polygon' x.style='low-poly' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/eagle' mv=True multirun=True

# pixel-art style:
## Darth vader
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='Darth vader with lightsaber. ultrarealistic.' x.style='pixelart' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/DarthVader' mv=True multirun=True

# painting style:
## van gogh
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='The image captures the essence of Vincent van Gogh. the tormented artist who created timeless masterpieces. Van Gogh is portrayed in a classic black and white photograph. which beautifully contrasts with the vibrant. colorful world he painted.' x.style='painting' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/VanGogh' mv=True multirun=True
## noche estrellada
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='Reinterpretation of the noche estrellada de Van Gogh. como si fuese un anime de studio gibbli' x.style='painting' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/NocheEstrellada' mv=True multirun=True

# sketch style:
## Dragon
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='A dragon flying in the sky. full body. minimal 2d line drawing. trending on artstation.' x.style='sketch' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.num_paths=128 result_path='./svgdreamer/dragon' mv=True multirun=True
## Lamborghini
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='A free-hand drawing of A speeding Lamborghini. black and white drawing.' x.style='sketch' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.num_paths=128 result_path='./svgdreamer/Lamborghini' mv=True multirun=True

# ink and wash style:
## Big Wild Goose Pagoda
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='Big Wild Goose Pagoda. ink style. Minimalist abstract art grayscale watercolor.' x.style='ink' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.num_paths=168 x.width=8 result_path='./svgdreamer/BigWildGoosePagoda' mv=True multirun=True
## villages
CUDA_VISIBLE_DEVICES=0 python svg_render.py x=svgdreamer prompt='Sepia ink wash landscape of ancient villages. Minimalist abstract art grayscale watercolor. on a white background.' x.style='ink' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.num_paths=256 x.width=6 result_path='./svgdreamer/InkVillages' mv=True multirun=True
```

**LIVE** vectorizes raster images (the emojis in original PNG format):

```shell
python svg_render.py x=live target='./data/simile.png'
```

**CLIPasso** synthesizes vectorized sketches from images:

**note:** first download the [U2Net model](https://huggingface.co/akhaliq/CLIPasso/blob/main/u2net.pth), and put the
model in `./checkpoint/u2net/u2net.pth`.

```shell
python svg_render.py x=clipasso target='./data/horse.png'
```

**CLIPDraw** synthesizes SVGs based on text prompts:

```shell
python svg_render.py x=clipdraw prompt='a photo of a cat'
```

**StyleCLIPDraw** synthesizes SVG based on a text prompt and a reference image:

```shell
python svg_render.py x=styleclipdraw prompt='a photo of a cat' target='./data/starry.png'
```

**CLIPFont** styles vector fonts according to text prompts:

```shell
python svg_render.py x=clipfont prompt='Starry Night by Vincent van gogh' target='./data/alphabet1.svg'
```

**VectorFusion** synthesizes SVGs in various styles based on text prompts:

```shell
# Iconography style
python svg_render.py x=vectorfusion prompt='a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. trending on artstation.'
# PixelArt style
python svg_render.py x=vectorfusion x.style='pixelart' prompt='a panda rowing a boat in a pond. pixel art. trending on artstation.'
# Sketch style
python svg_render.py x=vectorfusion x.style='sketch' prompt='a panda rowing a boat in a pond. minimal 2d line drawing. trending on artstation.'
```

**DiffSketcher** synthesizes vector sketches based on text prompts:

```shell
# DiffSketcher
python svg_render.py x=diffsketcher prompt='a photo of Sydney opera house' x.token_ind=5 seed=8019
# DiffSketcher, variable stroke width
python svg_render.py x=diffsketcher prompt='a photo of Sydney opera house' x.token_ind=5 x.optim_width=True seed=8019
# DiffSketcher RGBA version
python svg_render.py x=diffsketcher prompt='a photo of Sydney opera house' x.token_ind=5 x.optim_width=True x.optim_rgba=True x.optim_opacity=False seed=8019
# DiffSketcher + style transfer
python svg_render.py x=stylediffsketcher prompt='A horse is drinking water by the lake' x.token_ind=5 target='./data/starry.png' seed=998
```

**Word-As-Image** follow a text prompt to style a letter in a word:

```shell
# Inject the meaning of the word bunny into the 'Y' in the word 'BUNNY'
python svg_render.py x=wordasimage x.word='BUNNY' prompt='BUNNY' x.optim_letter='Y'
```

<h2 align="center">FAQ</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

<h2 align="center">TODO</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

- [x] integrated SVGDreamer.

<h2 align="center">Acknowledgement</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

The project is built based on the following repository:

[BachiLi/diffvg](https://github.com/BachiLi/diffvg), [huggingface/diffusers](https://github.com/huggingface/diffusers), [yael-vinker/CLIPasso](https://github.com/yael-vinker/CLIPasso), [ximinng/DiffSketcher](https://github.com/ximinng/DiffSketcher), [THUDM/ImageReward](https://github.com/THUDM/ImageReward), [advimman/lama](https://github.com/advimman/lama)

We gratefully thank the authors for their wonderful works.

<h2 align="center">Citation</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

If you use this code for your research, please cite the following work:

```
@article{xing2023svgdreamer,
    title={SVGDreamer: Text Guided SVG Generation with Diffusion Model},
    author={Xing, Ximing and Zhou, Haitao and Wang, Chuang and Zhang, Jing and Xu, Dong and Yu, Qian},
    journal={arXiv preprint arXiv:2312.16476},
    year={2023}
}
@inproceedings{xing2023diffsketcher,
    title={DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models},
    author={XiMing Xing and Chuang Wang and Haitao Zhou and Jing Zhang and Qian Yu and Dong Xu},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
    year={2023},
    url={https://openreview.net/forum?id=CY1xatvEQj}
}
```

<h2 align="center">Licence</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

This work is licensed under a **Mozilla Public License Version 2.0**.