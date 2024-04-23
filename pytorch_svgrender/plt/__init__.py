# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2023, XiMing Xing.
# License: MPL-2.0 License

from typing import AnyStr, BinaryIO, Union
from PIL import Image
import pathlib

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def plot_couple(input_1: torch.Tensor,
                input_2: torch.Tensor,
                step: int,
                output_dir: str,
                fname: Union[str, pathlib.Path, BinaryIO],  # file name
                prompt: str = '',  # text prompt as image tile
                pad_value: float = 0,
                dpi: int = 100):
    if input_1.shape != input_2.shape:
        raise ValueError("inputs and outputs must have the same dimensions")

    plt.figure()
    plt.subplot(1, 2, 1)  # nrows=1, ncols=2, index=1
    grid = make_grid(input_1, normalize=True, pad_value=pad_value)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title("Input")

    plt.subplot(1, 2, 2)  # nrows=1, ncols=2, index=2
    grid = make_grid(input_2, normalize=True, pad_value=pad_value)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"Rendering - {step} steps")

    def insert_newline(string, point=9):
        # split by blank
        words = string.split()
        if len(words) <= point:
            return string

        word_chunks = [words[i:i + point] for i in range(0, len(words), point)]
        new_string = "\n".join(" ".join(chunk) for chunk in word_chunks)
        return new_string

    plt.suptitle(insert_newline(prompt), fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fname}.png", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    plt.close()


def plot_img(inputs: torch.Tensor,
             output_dir: AnyStr,
             fname: Union[str, pathlib.Path, BinaryIO],  # file name
             pad_value: float = 0):
    assert torch.is_tensor(inputs), f"The input must be tensor type, but got {type(inputs)}"

    grid = make_grid(inputs, normalize=True, pad_value=pad_value)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    plt.imshow(ndarr)
    plt.axis("off")
    plt.tight_layout()
    plt.close()

    im = Image.fromarray(ndarr)
    im.save(f"{output_dir}/{fname}.png")


def plot_img_title(inputs: torch.Tensor,
                   title: str,
                   output_dir: AnyStr,
                   fname: Union[str, pathlib.Path, BinaryIO],  # file name
                   pad_value: float = 0,
                   dpi: int = 500):
    assert torch.is_tensor(inputs), f"The input must be tensor type, but got {type(inputs)}"

    grid = make_grid(inputs, normalize=True, pad_value=pad_value)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"{title}")
    plt.savefig(f"{output_dir}/{fname}.png", bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    plt.close()
