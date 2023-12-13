<h1 id="ptsvg" align="center">Pytorch-SVGRender</h1>

<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-3.10-or?logo=python" alt="pyhton">
    </a>
    <a href="http://mozilla.org/MPL/2.0/">
        <img src="https://img.shields.io/badge/license-MPL2.0-orange" alt="license">
    </a>
    <a href="/">
        <img src="https://img.shields.io/badge/website-Gitpage-yellow" alt="website">
    </a>
</p>

<p align="center">
    <a href="#recent-updates">Updates</a> •
    <a href="#installation">Installation</a> •
    <a href="#table-of-contents">Table of Contents</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#faq">FAQ</a> •
    <a href="#todo">TODO</a> •
    <a href="#acknowledgment">Acknowledgment</a> •
    <a href="#citation">Citation</a> •
    <a href="#licence">Licence</a>
</p>

Pytorch-SVGRender is the go-to library for state-of-the-art differentiable rendering methods for image vectorization.

<h2 align="center">Recent Updates</h2>

- [12/2023] We open-sourced the first version of Pytorch-SVGRender.

<h2 align="center">Installation</h2>

Create a new conda environment:

```shell
conda create --name svgrender python=3.10
conda activate svgrender
```

Install pytorch and the following libraries:

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install hydra-core omegaconf BeautifulSoup4
pip install freetype-py shapely svgutils
pip install opencv-python scikit-image matplotlib visdom wandb
pip install triton numba
pip install numpy scipy timm scikit-fmm einops
pip install accelerate transformers safetensors datasets
```

Install LaMa:

```shell
pip install easydict scikit-learn pytorch_lightning webdataset 
pip install albumentations==0.5.2
pip install kornia==0.5.0
pip install wldhx.yadisk-direct

cd lama
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

<h2 align="center">Table of Contents</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

### # DiffVG: Differentiable Vector Graphics Rasterization for Editing and Learning (`SIGGRAPH 2020`)

[[Project]](https://people.csail.mit.edu/tzumao/diffvg/) [[Paper]](https://cseweb.ucsd.edu/~tzli/diffvg/diffvg.pdf) [[Code]](https://github.com/BachiLi/diffvg)

DiffVG is a differentiable rasterizer for 2D vector graphics. **This repository is based on DiffVG.**

### CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders (`NIPS 2022`)

[[Paper]](https://arxiv.org/abs/2106.14843) [[Code]](https://github.com/kvfrans/clipdraw)

### CLIPFont: Texture Guided Vector WordArt Generation (`BMVC 2022`)

[[Paper]](https://bmvc2022.mpi-inf.mpg.de/0543.pdf) [[Code]](https://github.com/songyiren98/CLIPFont)

### StyleCLIPDraw: Coupling Content and Style in Text-to-Drawing Synthesis

[[Live]](https://slideslive.com/38970834/styleclipdraw-coupling-content-and-style-in-texttodrawing-synthesis?ref=account-folder-92044-folders) [[Paper]](https://arxiv.org/abs/2202.12362) [[Code]](https://github.com/pschaldenbrand/StyleCLIPDraw)

### LIVE: Towards Layer-wise Image Vectorization (`CVPR 2022`)

[[Project]](https://ma-xu.github.io/LIVE/) [[Paper]](https://ma-xu.github.io/LIVE/index_files/CVPR22_LIVE_main.pdf) [[Code]](https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization)

### CLIPasso: Semantically-Aware Object Sketching (`SIGGRAPH 2022`)

[[Project]](https://clipasso.github.io/clipasso/) [[Paper]](https://arxiv.org/abs/2202.05822) [[Code]](https://github.com/yael-vinker/CLIPasso)

### CLIPascene: Scene Sketching with Different Types and Levels of Abstraction (`ICCV 2023`)

[[Project]](https://clipascene.github.io/CLIPascene/) [[Paper]](https://arxiv.org/abs/2211.17256) [[Code]](https://github.com/yael-vinker/SceneSketch)

### VectorFusion: Text-to-SVG by Abstracting Pixel-Based Diffusion Models (`CVPR 2023`)

[[Project]](https://vectorfusion.github.io/) [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Jain_VectorFusion_Text-to-SVG_by_Abstracting_Pixel-Based_Diffusion_Models_CVPR_2023_paper.pdf)

### # DiffSketcher (`NIPS 2023`)

[[Project]](https://ximinng.github.io/DiffSketcher-project/) [[Paper]](https://arxiv.org/abs/2306.14685) [[Code]](https://github.com/ximinng/DiffSketcher)

### # Word-As-Image for Semantic Typography (`SIGGRAPH 2023`)

[[Project]](https://wordasimage.github.io/Word-As-Image-Page/) [[Paper]](https://arxiv.org/abs/2303.01818) [[Code]](https://github.com/Shiriluz/Word-As-Image)

<h2 align="center">Quickstart</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

<h2 align="center">FAQ</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

<h2 align="center">TODO</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

- [ ] integrated the [Hydra](https://hydra.cc/docs/intro/), [repo link](https://github.com/facebookresearch/hydra)

<h2 align="center">Acknowledgement</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

The project is built based on the following repository:

- [BachiLi/diffvg](https://github.com/BachiLi/diffvg)
- [ma-xu/LIVE](https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization)
- [huggingface/diffusers](https://github.com/huggingface/diffusers)
- [yael-vinker/CLIPasso](https://github.com/yael-vinker/CLIPasso)

We gratefully thank the authors for their wonderful works.

<h2 align="center">Citation</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

If you use this code for your research, please cite the following work:

```
@inproceedings{
    xing2023diffsketcher,
    title={DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models},
    author={XiMing Xing and Chuang Wang and Haitao Zhou and Jing Zhang and Qian Yu and Dong Xu},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=CY1xatvEQj}
}
```

<h2 align="center">Licence</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

This work is licensed under a **Mozilla Public License Version 2.0**.