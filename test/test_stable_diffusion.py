# -*- coding: utf-8 -*-
# Author: ximing
# Description: test_sd_models
# Copyright (c) 2024, XiMing Xing.
# License: MPL-2.0 License

import random
from pathlib import Path
from diffusers.utils import load_image, make_image_grid
from accelerate.utils import set_seed


def test_SDXL():
    from diffusers import AutoPipelineForText2Image, StableDiffusionXLImg2ImgPipeline
    import torch

    set_seed(seed=random.randint(0, 9999999))

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")
    pipeline_text2image.enable_xformers_memory_efficient_attention()

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        local_files_only=True,
    ).to("cuda")
    refiner.enable_xformers_memory_efficient_attention()

    # prompt = "A variety of vector graphics. vector art."
    # prompt = "unicorn, Die-cut sticker, Cute kawaii flower character sticker, white background, illustration minimalism, vector, pastel colors"
    prompt = "DigiArtist holds a shiny SVG paintbrush, Die-cut sticker, Cute kawaii character sticker, 3d blender render, white background, illustration minimalism, vector, pastel colors, physically based rendering"
    # prompt = "the batman, Die-cut sticker, Cute kawaii character sticker, white background, illustration minimalism, vector, pastel colors"

    save_path = Path("./test/sdxl-DigiArtist-3")
    save_path.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        image = pipeline_text2image(prompt=prompt).images[0]

        refined_image = refiner(
            prompt=prompt,
            num_inference_steps=60,
            denoising_start=0.8,
            image=image,
        ).images[0]

        img = make_image_grid([image], rows=1, cols=1)
        img.save(save_path / f'base_{i}.png')
        img = make_image_grid([refined_image], rows=1, cols=1)
        img.save(save_path / f'refined_{i}.png')


if __name__ == '__main__':
    # python test/test_stable_diffusion.py
    test_SDXL()
