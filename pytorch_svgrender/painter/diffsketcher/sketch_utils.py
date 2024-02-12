# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def plt_triplet(
        photos: torch.Tensor,
        sketch: torch.Tensor,
        style: torch.Tensor,
        step: int,
        prompt: str,
        output_dir: str,
        fname: str,  # file name
        dpi: int = 300
):
    if photos.shape != sketch.shape:
        raise ValueError("photos and sketch must have the same dimensions")

    plt.figure()
    plt.subplot(1, 3, 1)  # nrows=1, ncols=3, index=1
    grid = make_grid(photos, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title("Generated sample")

    plt.subplot(1, 3, 2)  # nrows=1, ncols=3, index=2
    # style = (style + 1) / 2
    grid = make_grid(style, normalize=False, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"Style")

    plt.subplot(1, 3, 3)  # nrows=1, ncols=3, index=2
    # sketch = (sketch + 1) / 2
    grid = make_grid(sketch, normalize=False, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"Rendering result - {step} steps")

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


def plt_attn(attn: np.array,
             threshold_map: np.array,
             inputs: torch.Tensor,
             inds: np.array,
             output_path: str):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input img")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("attn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / \
                     (threshold_map.max() - threshold_map.min())
    plt.imshow(np.nan_to_num(threshold_map_), interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
