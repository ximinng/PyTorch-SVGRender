# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing

from setuptools import setup, find_packages

setup(
    name='PyTorch-SVGRender',
    packages=find_packages(exclude=["test*", "docs", "examples"]),
    version='1.0.0',
    license='Mozilla Public License Version 2.0',
    description='SVG Differentiable Rendering: Generating vector graphics using neural networks.',
    author='Ximing Xing, Juncheng Hu et al.',
    author_email='ximingxing@gmail.com',
    url='https://github.com/ximinng/PyTorch-SVGRender',
    long_description_content_type='text/markdown',
    keywords=[
        'artificial intelligence',
        'AIGC',
        'generative models',
        'SVG generation',
    ],
    install_requires=[
        'hydra-core==1.3.2',  # configuration processor
        'omegaconf==2.3.0',  # YAML processor
        'accelerate==0.20.3',  # HuggingFace - pytorch distributed configuration
        'diffusers==0.20.2',  # HuggingFace - diffusion models
        'transformers==4.30.2',  # HuggingFace - transformers
        'datasets==2.13.1',  # #HuggingFace - datasets
        'safetensors==0.3.1',
        'xformers',  # speed up attn compute
        'einops==0.6.1',
        'pillow',  # keep the PIL.Image.Resampling deprecation away,
        'imageio-ffmpeg==0.4.8'
        'torch>=1.13.1',
        'torchvision>=0.14.1',
        'tensorboard==2.14.0',
        'triton==2.0.0.post1',
        'numba==0.57.1',
        'tqdm',  # progress bar
        'ftfy==6.1.1',
        'regex==2023.6.3',
        'timm==0.6.13',  # computer vision models
        "numpy==1.24.4",
        'scikit-learn==1.3.2',
        'scikit-fmm==2023.4.2',
        'scipy==1.10.1',
        'scikit-image==0.20.0',
        'pytorch-lightning==2.1.0',
        'matplotlib==3.7.1',
        'visdom=0.2.4',
        'wandb==0.15.8',  # weights & Biases
        'opencv-python==4.8.0.74',  # cv2
        'BeautifulSoup4==4.12.2',
        'freetype-py',  # font
        'shapely==2.0.1',  # SVG
        'svgwrite==1.4.3',
        'svgutils==0.3.4',
        'svgpathtools==1.6.1',
        'fairscale=0.4.13'  # ImageReward
    ],
    dependency_links=[
        "clip @ git+https://github.com/openai/CLIP.git",
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)
