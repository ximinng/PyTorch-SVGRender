# -*- coding: utf-8 -*-
import pathlib
from typing import Union, Optional, List, Tuple, Dict, Text, BinaryIO
from PIL import Image

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .seq_aligner import get_word_inds


def text_under_image(image: np.ndarray,
                     text: str,
                     text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(
        images: Union[np.ndarray, List[np.ndarray]],
        num_rows: int = 1,
        offset_ratio: float = 0.02,
        save_image: bool = False,
        fp: Union[Text, pathlib.Path, BinaryIO] = None,
) -> np.ndarray:
    if save_image:
        assert fp is not None

    if isinstance(images, list):
        images = np.concatenate(images, axis=0)

    if isinstance(images, np.ndarray) and images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images] if not isinstance(images, list) else images
        num_empty = len(images) % num_rows

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    # Calculate the composite image
    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = int(np.ceil(num_items / num_rows))  # count the number of columns
    image_h = h * num_rows + offset * (num_rows - 1)
    image_w = w * num_cols + offset * (num_cols - 1)
    assert image_h > 0, "Invalid image height: {} (num_rows={}, offset_ratio={}, num_items={})".format(
        image_h, num_rows, offset_ratio, num_items)
    assert image_w > 0, "Invalid image width: {} (num_cols={}, offset_ratio={}, num_items={})".format(
        image_w, num_cols, offset_ratio, num_items)
    image_ = np.ones((image_h, image_w, 3), dtype=np.uint8) * 255

    # Ensure that the last row is filled with empty images if necessary
    if len(images) % num_cols > 0:
        empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
        num_empty = num_cols - len(images) % num_cols
        images += [empty_images] * num_empty

    for i in range(num_rows):
        for j in range(num_cols):
            k = i * num_cols + j
            if k >= num_items:
                break
            image_[i * (h + offset): i * (h + offset) + h, j * (w + offset): j * (w + offset) + w] = images[k]

    pil_img = Image.fromarray(image_)
    if save_image:
        pil_img.save(fp)
    return pil_img


def update_alpha_time_word(alpha,
                           bounds: Union[float, Tuple[float, float]],
                           prompt_ind: int,
                           word_inds: Optional[torch.Tensor] = None):
    if isinstance(bounds, float):
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer,
                                   max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words
