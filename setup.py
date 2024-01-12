# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing

from setuptools import setup, find_packages

setup(
    name='PyTorch-SVGRender',
    packages=find_packages(exclude=["test*"]),
    version='1.0.0',
    license='Mozilla Public License Version 2.0',
    description='SVG Differentiable Rendering: Generating vector graphics using neural networks.',
    author='XiMing Xing et al.',
    author_email='ximingxing@gmail.com',
    url='https://github.com/ximinng/PyTorch-SVGRender',
    long_description_content_type='text/markdown',
    keywords=[
        'artificial intelligence',
        'AIGC',
        'generative models',
        'SVG',
    ],
    install_requires=[
        'hydra-core',  # configuration processor
        'omegaconf',  # YAML processor
        'accelerate',  # Hugging Face - pytorch distributed configuration
        'diffusers==0.20.2',  # Hugging Face - diffusion models
        'transformers',  # Hugging Face - transformers
        'safetensors',
        'xformers',
        'einops',
        'pillow',
        'torch>=1.13.1',
        'torchvision',
        'tensorboard',
        'torchmetrics',
        'triton',
        'numba',
        'tqdm',  # progress bar
        'ftfy',
        'regex',
        'timm',  # computer vision models
        "numpy",  # numpy
        'scikit-learn',
        'scikit-fmm',
        'scipy',
        'scikit-image',
        'Pillow',  # keep the PIL.Image.Resampling deprecation away,
        'matplotlib',
        'visdom',
        'wandb',  # weights & Biases
        'opencv-python',  # cv2
        'BeautifulSoup4',
        'freetype-py',  # font
        'shapely',  # SVG
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
