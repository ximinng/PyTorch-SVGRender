# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2023, XiMing Xing.
# License: MPL-2.0 License

from typing import AnyStr

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def plot_couple(input_1: torch.Tensor,
                input_2: torch.Tensor,
                step: int,
                output_dir: str,
                fname: str,  # file name
                prompt: str = '',  # text prompt as image tile
                dpi: int = 300):
    if input_1.shape != input_2.shape:
        raise ValueError("inputs and outputs must have the same dimensions")

    plt.figure()
    plt.subplot(1, 2, 1)  # nrows=1, ncols=2, index=1
    grid = make_grid(input_1, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title("Input")

    plt.subplot(1, 2, 2)  # nrows=1, ncols=2, index=2
    grid = make_grid(input_2, normalize=True, pad_value=2)
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
    plt.savefig(f"{output_dir}/{fname}.png", dpi=dpi)
    plt.close()


def plot_img(inputs: torch.Tensor,
             output_dir: AnyStr,
             fname: str,  # file name
             dpi: int = 100):
    assert torch.is_tensor(inputs), f"The input must be tensor type, but got {type(inputs)}"

    grid = make_grid(inputs, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fname}.png", dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_img_title(inputs: torch.Tensor,
                   title: str,
                   output_dir: AnyStr,
                   fname: str,  # file name
                   dpi: int = 500):
    assert torch.is_tensor(inputs), f"The input must be tensor type, but got {type(inputs)}"

    grid = make_grid(inputs, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"{title}")
    plt.savefig(f"{output_dir}/{fname}.png", dpi=dpi)
    plt.close()
