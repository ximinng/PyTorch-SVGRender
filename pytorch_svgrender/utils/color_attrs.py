# -*- coding: utf-8 -*-
# Author: ximing
# Description: shape_group
# Copyright (c) 2023, XiMing Xing.
# License: MIT License

from typing import Tuple

import torch
from matplotlib import colors


def init_tensor_with_rgb(
        rgb: Tuple[float, float, float],
        b: int,
        w: int,
        h: int,
        norm: bool = False
):
    """
    Initializes a PyTorch tensor with the specified RGB values. The tensor has shape (b, 3, w, h).

    Args:
      rgb: RGB values, shape (3,)
      b: Batch size
      w: Width
      h: Height
      norm: normalize the tensor to range [0, 1]

    Examples:
        >>>  rgb = (0.5, 0.2, 0.1)   # Specify RGB values
        >>>  tensor = init_tensor_with_rgb(rgb, 1, 100, 100, norm=False)   # Initialize tensor

    Returns:
      Initialized tensor
    """

    # Convert RGB values to tensor
    rgb = torch.tensor(rgb, dtype=torch.float)

    # Create tensor
    tensor = torch.zeros((b, 3, w, h), dtype=torch.float)

    # Assign RGB values to tensor
    tensor[:, 0] = rgb[0]
    tensor[:, 1] = rgb[1]
    tensor[:, 2] = rgb[2]

    if norm:
        tensor = tensor / 255.

    return tensor


def init_tensor_with_color(
        color: str,
        b: int,
        w: int,
        h: int,
        norm: bool = True
):
    """
    Initializes a PyTorch tensor with the specified RGB values. The tensor has shape (b, 3, w, h).

    Args:
      color:
      b: Batch size
      w: Width
      h: Height
      norm: normalize the tensor to range [0, 1]

    Examples:
        >>>  color = '#B0A695'   # Specify RGB values
        >>>  tensor = init_tensor_with_rgb(color, 1, 100, 100)   # Initialize tensor

    Returns:
      Initialized tensor
    """

    rgb = get_rgb_from_color(color)

    # Convert RGB values to tensor
    rgb = torch.tensor(rgb, dtype=torch.float)

    # Create tensor
    tensor = torch.zeros((b, 3, w, h), dtype=torch.float)

    # Assign RGB values to tensor
    tensor[:, 0] = rgb[0]
    tensor[:, 1] = rgb[1]
    tensor[:, 2] = rgb[2]

    return tensor


def hex_to_rgb(hex_code):
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return (r, g, b)


def get_rgb_from_color(color: str):
    # get the corresponding RGB value based on the color
    if color.startswith('#'):
        color = color.split('#')[1]
        rgb = hex_to_rgb(color)
        rgb = [c / 255. for c in rgb]  # to [0, 1]
    elif color in colors.cnames:
        rgb = colors.to_rgb(color)
    else:
        rgb = color
    return rgb


if __name__ == "__main__":
    color = '#B0A695'

    rgb = get_rgb_from_color(color)

    print(rgb)
