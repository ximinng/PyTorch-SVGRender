<h1 id="ptsvg" align="center">Pytorch-SVGRender</h1>

<p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10-or?logo=python" alt="pyhton"></a>
    <a href="http://mozilla.org/MPL/2.0/"><img src="https://img.shields.io/badge/license-MPL2.0-orange" alt="license"></a>
    <a href="https://ximinng.github.io/PyTorch-SVGRender-project/"><img src="https://img.shields.io/badge/website-Gitpage-yellow" alt="website"></a>
    <a href="https://pytorch-svgrender.readthedocs.io/en/latest/index.html"><img src="https://img.shields.io/badge/docs-readthedocs-purple" alt="docs"></a>
    <a href="https://huggingface.co/SVGRender"><img src="https://img.shields.io/badge/Group-Join%20Us-FC819E" alt="group"></a>
</p>

<div align="center">
<img src="./assets/cover1.png" height="70%" width="auto" alt="Pytorch-SVGRender">
<p><strong>Pytorch-SVGRender: </strong>The go-to library for differentiable rendering methods for SVG generation.</p>
</div>
<p align="center">
    <a href="#recent-updates">Updates</a> •
    <a href="#table-of-contents">Table of Contents</a> •
    <a href="#installation">Installation</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#faq">FAQ</a> •
    <a href="#todo">TODO</a> •
    <a href="#contribution">Contribution</a> •
    <a href="#acknowledgement">Acknowledgment</a> •
    <a href="#citation">Citation</a> •
    <a href="#licence">Licence</a>
</p>

<h2 align="center">🔥 Recent Updates</h2>

- [12/2023] 🔥 We open-sourced Pytorch-SVGRender V1.0.

<h2 align="center">💡 Table of Contents</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

### 1. Image Vectorization

- DiffVG: Differentiable Vector Graphics Rasterization for Editing and Learning (`SIGGRAPH 2020`)

  [[Project]](https://people.csail.mit.edu/tzumao/diffvg/) [[Paper]](https://cseweb.ucsd.edu/~tzli/diffvg/diffvg.pdf) [[Code]](https://github.com/BachiLi/diffvg)

  DiffVG is a differentiable rasterizer for 2D vector graphics. **This repository is heavily based on DiffVG.**

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

- SVGDreamer: Text Guided SVG Generation with Diffusion Model (`CVPR 2024`)

  [[Project]](https://ximinng.github.io/SVGDreamer-project/) [[Paper]](https://arxiv.org/abs/2312.16476) [[Code]](https://github.com/ximinng/SVGDreamer) [[Blog]](https://huggingface.co/blog/xingxm/svgdreamer/)

<h2 align="center">⚙️ Installation</h2>

You can follow the steps below to quickly get up and running with PyTorch-SVGRender.
These steps will let you run quick inference locally.

In the top level directory run,

```bash
sh script/install.sh
```

Note: Make sure that the script file has execution **permissions** (you can give them using `chmod +x script.sh`), and
then run the script.

or using docker,

```shell
docker run --name svgdreamer --gpus all -it --ipc=host ximingxing/svgrender:v1 /bin/bash
```

For more information, please refer to
the [Install.md](https://github.com/ximinng/PyTorch-SVGRender/blob/main/Install.md).

<h2 align="center">👩‍🎨🎨 Quickstart</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

**For more information, [read the docs](https://pytorch-svgrender.readthedocs.io/en/latest/index.html).**

### 1. Basic Usage

**DiffVG** vectorizes any raster images:

```shell
python svg_render.py x=diffvg target='./data/fallingwater.png'
# change 'num_paths' and 'num_iter' for better results
python svg_render.py x=diffvg target='./data/fallingwater.png' x.num_paths=512 x.num_iter=2000
```

**LIVE** vectorizes the raster emojis images (in original PNG format):

```shell
python svg_render.py x=live target='./data/simile.png'
# change 'num_paths' and 'schedule_each' for better results
python svg_render.py x=live target='./data/simile.png' x.num_paths=5 x.schedule_each=1
```

**CLIPasso** synthesizes vectorized sketches from images:

**note:** first download the U2Net model `sh script/download_u2net.sh`.

```shell
python svg_render.py x=clipasso target='./data/horse.png'
```

**CLIPascene** synthesizes vectorized sketches from images:

**note:** first download the U2Net model `sh script/download_u2net.sh`, and make sure the `./data/background` folder and
the `./data/scene` folder exist with target images.

```shell
python svg_render.py x=clipascene target='ballerina.png'
```

**CLIPDraw** synthesizes SVGs based on text prompts:

```shell
python svg_render.py x=clipdraw "prompt='a photo of a cat'"
```

**StyleCLIPDraw** synthesizes SVG based on a text prompt and a reference image:

```shell
python svg_render.py x=styleclipdraw "prompt='a photo of a cat'" target='./data/starry.png'
```

**CLIPFont** styles vector fonts according to text prompts:

```shell
python svg_render.py x=clipfont "prompt='Starry Night by Vincent van gogh'" target='./data/alphabet1.svg'
```

---

> Because the following methods rely on stable diffusion, add `diffuser.download=True` to the command the **first time**
> you
> run the script.

**SVGDreamer** generates various styles of SVG based on text prompts. It supports the use of six vector primitives,
including Iconography, Sketch, Pixel Art, Low-Poly, Painting, and Ink and Wash.

```shell
# primitive: iconography
## 1. German shepherd
python svg_render.py x=svgdreamer "prompt='A colorful German shepherd in vector art. tending on artstation.'" save_step=50 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/GermanShepherd'
## 2. sydney opera house
python svg_render.py x=svgdreamer "prompt='Sydney opera house. oil painting. by Van Gogh'" save_step=50 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.num_paths=512 result_path='./svgdreamer/SydneyOperaHouse'
# primitive: low-ploy
python svg_render.py x=svgdreamer "prompt='A picture of a bald eagle. low-ploy. polygon'" x.style='low-poly' save_step=50 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.grid=30 x.guidance.num_iter=1000 result_path='./svgdreamer/BaldEagle'
# primitive: pixel-art
python svg_render.py x=svgdreamer "prompt='Darth vader with lightsaber. ultrarealistic. pixelart. trending on artstation.'" x.style='pixelart' save_step=50 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.guidance.num_iter=1000 result_path='./svgdreamer/DarthVader'
# primitive: painting
python svg_render.py x=svgdreamer "prompt='self portrait of Van Gogh. oil painting. cmyk portrait. multi colored. defiant and beautiful. cmyk. expressive eyes.'" x.style='painting' save_step=50 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.guidance.t_schedule='randint' x.num_paths=1500 result_path='./svgdreamer/VanGogh_portrait'
# primitive: sketch
python svg_render.py x=svgdreamer "prompt='A free-hand drawing of A speeding Lamborghini. black and white drawing.'" x.style='sketch' save_step=50 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.guidance.t_schedule='randint' x.num_paths=128 result_path='./svgdreamer/Lamborghini'
# primitive: ink and wash
python svg_render.py x=svgdreamer "prompt='Big Wild Goose Pagoda. ink style. Minimalist abstract art grayscale watercolor.'" x.style='ink' save_step=50 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.guidance.t_schedule='randint' x.num_paths=128 x.width=6 result_path='./svgdreamer/BigWildGoosePagoda'
```

**VectorFusion** synthesizes SVGs in various styles based on text prompts:

```shell
# Iconography style
python svg_render.py x=vectorfusion x.style='iconography' "prompt='a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. trending on artstation.'"
# PixelArt style
python svg_render.py x=vectorfusion x.style='pixelart' "prompt='a panda rowing a boat in a pond. pixel art. trending on artstation.'"
# Sketch style
python svg_render.py x=vectorfusion x.style='sketch' "prompt='a panda rowing a boat in a pond. minimal 2d line drawing. trending on artstation.'"
```

Following SVGDreamer, we've added three additional styles (`Paining`, `Ink and Wash` and `low-ploy`) to VectorFusion.

**DiffSketcher** synthesizes vector sketches based on text prompts:

```shell
# DiffSketcher
python svg_render.py x=diffsketcher "prompt='a photo of Sydney opera house'" x.token_ind=5 seed=8019
# DiffSketcher, variable stroke width
python svg_render.py x=diffsketcher "prompt='a photo of Sydney opera house'" x.token_ind=5 x.optim_width=True seed=8019
# DiffSketcher RGBA version
python svg_render.py x=diffsketcher "prompt='a photo of Sydney opera house'" x.token_ind=5 x.optim_width=True x.optim_rgba=True x.optim_opacity=False seed=8019
# DiffSketcher + style transfer
python svg_render.py x=stylediffsketcher "prompt='The French Revolution. highly detailed. 8k. ornate. intricate. cinematic. dehazed. atmospheric. oil painting. by Van Gogh'" x.token_ind=4 x.num_paths=2000 target='./data/starry.png' seed=876809
```

**Word-As-Image** follow a text prompt to style a letter in a word:

```shell
# Inject the meaning of the word bunny into the 'Y' in the word 'BUNNY'
python svg_render.py x=wordasimage x.word='BUNNY' prompt='BUNNY' x.optim_letter='Y'
# Change font: 'LuckiestGuy-Regular', default: 'KaushanScript-Regular'
python svg_render.py x=wordasimage x.word='DRAGONFLY' prompt='Dragonfly' x.optim_letter='Y' x.font='LuckiestGuy-Regular'
```

### 2. SDS Loss based Approach

This is achieved by utilizing a pretrained text-to-image diffusion model as a strong image prior to supervise the
training of the PyDiffVG, enabling rendering SVG alignment with the text. This remarkable capability is fundamentally
grounded in the use of Score Distillation Sampling (SDS). SDS acts as the core mechanism that lifts raster images from
diffusion models to the SVG domain, enabling the training of SVG parameters without images.
This includes the methods VectorFusion, DiffSketcher and SVGDreamer.

We only compare the performance of SDS, which means that no other loss is used:

```shell
# SDS loss
python svg_render.py x=vectorfusion "prompt='a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. trending on artstation.'"
# Input Augmentation SDS loss (LSDS loss)
python svg_render.py x=vectorfusion x.style='sketch' "prompt='an elephant. minimal 2d line drawing. trending on artstation.'" x.skip_live=True
# Input Augmentation SDS loss (ASDS loss)
python svg_render.py x=diffsketcher "prompt='an elephant. minimal 2d line drawing. trending on artstation.'" x.token_ind=2 x.sds.grad_scale=1 x.sds.num_aug=4 x.clip.vis_loss=0 x.perceptual.coeff=0 x.opacity_delta=0.3 
# Vectorized Particle-based Score Distillation (VPSD loss)
python svg_render.py x=svgdreamer "prompt='a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. trending on artstation.'" save_step=60 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 
```

<h2 align="center">❓ FAQ</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

- Q: Where can I get more scripts and visualizations?
- A: Check the [pytorch-svgrender.readthedocs.io](https://pytorch-svgrender.readthedocs.io/en/latest/index.html).

- Q: An error says HuggingFace cannot find the model in the disk cache.
- A: Add *`diffuser.download=True`* to the command for downloading model checkpoints the **first time** you run the
  script.

- Q: It says xFormers is not built with CUDA support or xFormers cannot load C++/CUDA extensions.
- A: You need to install xFormers again using the command *`pip install --pre -U xformers`* instead of the conda one.

<h2 align="center">🗒 TODO</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

- [x] SVGDreamer supports fp16 optimization.
- [x] integrated SVGDreamer.

<h2 align="center">🤝 Contribution</h2>

- How to Contribute to the PyTorch-SVGRender Community?
- We greatly appreciate your contributions to the [PyTorch-SVGRender community](https://huggingface.co/SVGRender) and
  helping us make it even better than it is now! For more details, please refer to
  the [Contribution Guidelines](https://github.com/ximinng/PyTorch-SVGRender/blob/main/Contribution.md).

  **Remember** to use `test/test_svgrender.py` to test your code before committing it.

<h2 align="center">💘 Acknowledgement</h2>
<p align="right"><a href="#ptsvg"><sup>▴ Back to top</sup></a></p>

The project is built based on the following repository:

[BachiLi/diffvg](https://github.com/BachiLi/diffvg),
[huggingface/diffusers](https://github.com/huggingface/diffusers),
[threestudio-project/threestudio](https://github.com/threestudio-project/threestudio),
[yael-vinker/CLIPasso](https://github.com/yael-vinker/CLIPasso),
[ximinng/DiffSketcher](https://github.com/ximinng/DiffSketcher),
[THUDM/ImageReward](https://github.com/THUDM/ImageReward),
[advimman/lama](https://github.com/advimman/lama)

We gratefully thank the authors for their wonderful works.

<h2 align="center">📚 Citation</h2>
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
