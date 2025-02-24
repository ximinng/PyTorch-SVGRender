# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing

import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info


class CustomInstallCommand(install):
    """Custom installation command to handle dependencies and CUDA setup"""

    def run(self):
        self.install_dependencies()
        install.run(self)

    def install_dependencies(self):
        try:
            self._check_requirements()
            self._setup_cuda()
            self._install_pytorch()
            self._install_python_deps()
            self._install_diffvg()
        except Exception as e:
            print(f"\033[91mError during installation: {str(e)}\033[0m")
            sys.exit(1)

    def _check_requirements(self):
        """Check system requirements"""
        print("\033[92mChecking system requirements...\033[0m")

        # Check conda
        try:
            import conda
        except ImportError:
            raise RuntimeError("Conda is required. Please install Conda first.")

        # Check git
        if subprocess.call(['which', 'git'], stdout=subprocess.PIPE) != 0:
            raise RuntimeError("Git is required. Please install Git first.")

    def _setup_cuda(self):
        """Check CUDA availability"""
        print("\033[92mChecking CUDA availability...\033[0m")

        try:
            subprocess.check_output(['nvidia-smi'])
            self.cuda_available = True
            print("CUDA is available")
        except:
            self.cuda_available = False
            print("\033[93mCUDA not available. Installing CPU-only version.\033[0m")

    def _install_pytorch(self):
        """Install PyTorch and related packages"""
        print("\033[92mInstalling PyTorch...\033[0m")

        if self.cuda_available:
            pytorch_cmd = [
                'conda', 'install', '-y',
                'pytorch==1.12.1',
                'torchvision==0.13.1',
                'torchaudio==0.12.1',
                'cudatoolkit=11.3',
                '-c', 'pytorch'
            ]
        else:
            pytorch_cmd = [
                'conda', 'install', '-y',
                'pytorch==1.12.1',
                'torchvision==0.13.1',
                'torchaudio==0.12.1',
                'cpuonly',
                '-c', 'pytorch'
            ]

        subprocess.check_call(pytorch_cmd)

        try:
            subprocess.check_call([
                'conda', 'install', '-y', 'xformers',
                '-c', 'xformers'
            ])
        except:
            print("\033[93mWarning: Failed to install xformers\033[0m")

    def _install_python_deps(self):
        """Install Python dependencies"""
        print("\033[92mInstalling Python dependencies...\033[0m")

        pip_packages = [
            'hydra-core', 'omegaconf',
            'freetype-py', 'shapely', 'svgutils', 'cairosvg',
            'opencv-python', 'scikit-image', 'matplotlib', 'visdom', 'wandb', 'beautifulsoup4',
            'triton', 'numba',
            'numpy', 'scipy', 'scikit-fmm', 'einops', 'timm', 'fairscale==0.4.13',
            'accelerate', 'transformers', 'safetensors', 'datasets',
            'easydict', 'scikit-learn', 'pytorch_lightning==2.1.0', 'webdataset',
            'ftfy', 'regex', 'tqdm',
            'diffusers==0.20.2',
            'svgwrite', 'svgpathtools', 'cssutils', 'torch-tools'
        ]

        for package in pip_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except:
                print(f"\033[93mWarning: Failed to install {package}\033[0m")

        # Install CLIP
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                'git+https://github.com/openai/CLIP.git'
            ])
        except:
            print("\033[93mWarning: Failed to install CLIP\033[0m")

    def _install_diffvg(self):
        """Install DiffVG"""
        print("\033[92mInstalling DiffVG...\033[0m")

        if not os.path.exists('diffvg'):
            subprocess.check_call(['git', 'clone', 'https://github.com/BachiLi/diffvg.git'])

        os.chdir('diffvg')
        subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])

        # Install system dependencies on Linux
        if sys.platform.startswith('linux'):
            try:
                subprocess.check_call([
                    'sudo', 'apt', 'update'
                ])
                subprocess.check_call([
                    'sudo', 'apt', 'install', '-y',
                    'cmake', 'ffmpeg', 'build-essential',
                    'libjpeg-dev', 'libpng-dev', 'libtiff-dev'
                ])
            except:
                print("\033[93mWarning: Failed to install system dependencies\033[0m")

        # Install conda dependencies
        subprocess.check_call(['conda', 'install', '-y', '-c', 'anaconda', 'cmake'])
        subprocess.check_call(['conda', 'install', '-y', '-c', 'conda-forge', 'ffmpeg'])

        # Build and install DiffVG
        subprocess.check_call([sys.executable, 'setup.py', 'install'])
        os.chdir('..')


setup(
    name='PyTorch-SVGRender',
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Base dependencies will be handled by custom install command
    ],
    python_requires=">=3.7",
    cmdclass={
        'install': CustomInstallCommand,
    },
    # Metadata
    author='Ximing Xing, Juncheng Hu et al.',
    author_email='ximingxing@gmail.com',
    description='SVG Differentiable Rendering: Generating vector graphics using neural networks.',
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    keywords=[
        'Artificial Intelligence',
        'AIGC',
        'Generative Models',
        'SVG Generation',
    ],
    url='https://github.com/ximinng/PyTorch-SVGRender',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
