# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2023, XiMing Xing.
# License: MPL-2.0 License

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def plot_batch(inputs: torch.Tensor,
               outputs: torch.Tensor,
               step: int,
               save_path: str,
               name: str,
               prompt: str = '',
               dpi: int = 300):
    if inputs.shape != outputs.shape:
        raise ValueError("inputs and outputs must have the same dimensions")

    plt.figure()
    plt.subplot(1, 2, 1)  # nrows=1, ncols=2, index=1
    grid = make_grid(inputs, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title("Input")

    plt.subplot(1, 2, 2)  # nrows=1, ncols=2, index=2
    grid = make_grid(outputs, normalize=True, pad_value=2)
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
    plt.savefig(f"{save_path}/{name}.png", dpi=dpi)
    plt.close()


def log_input(inputs, output_dir, output_prefix="input", dpi=100):
    grid = make_grid(inputs, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"{output_dir}/{output_prefix}.png", dpi=dpi)
    plt.close()


def plt_tensor_img(tensor, title, save_path, name, dpi=500):
    grid = make_grid(tensor, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"{title}")
    plt.savefig(f"{save_path}/{name}.png", dpi=dpi)
    plt.close()
