# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import shutil
from PIL import Image
from pathlib import Path

import torch
from torchvision import transforms
import clip
from tqdm.auto import tqdm
import numpy as np

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.painter.style_clipdraw import (
    Painter, PainterOptimizer, VGG16Extractor, StyleLoss, sample_indices
)
from pytorch_svgrender.plt import plot_img, plot_couple


class StyleCLIPDrawPipeline(ModelState):

    def __init__(self, args):
        logdir_ = f"sd{args.seed}" \
                  f"-P{args.x.num_paths}" \
                  f"-style{args.x.style_strength}" \
                  f"-n{args.x.num_aug}"
        super().__init__(args, log_path_suffix=logdir_)

        # create log dir
        self.png_logs_dir = self.result_path / "png_logs"
        self.svg_logs_dir = self.result_path / "svg_logs"
        if self.accelerator.is_main_process:
            self.png_logs_dir.mkdir(parents=True, exist_ok=True)
            self.svg_logs_dir.mkdir(parents=True, exist_ok=True)

        # make video log
        self.make_video = self.args.mv
        if self.make_video:
            self.frame_idx = 0
            self.frame_log_dir = self.result_path / "frame_logs"
            self.frame_log_dir.mkdir(parents=True, exist_ok=True)

        self.clip, self.tokenize_fn = self.init_clip()

        self.style_extractor = VGG16Extractor(space="normal").to(self.device)
        self.style_loss = StyleLoss()

    def init_clip(self):
        model, _ = clip.load('ViT-B/32', self.device, jit=False)
        return model, clip.tokenize

    def drawing_augment(self, image):
        augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        # image augmentation transformation
        img_augs = []
        for n in range(self.x_cfg.num_aug):
            img_augs.append(augment_trans(image))
        im_batch = torch.cat(img_augs)
        # clip visual encoding
        image_features = self.clip.encode_image(im_batch)

        return image_features

    def style_file_preprocess(self, style_file):
        process_comp = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
            transforms.Lambda(lambda t: (t + 1) / 2),
        ])
        style_file = process_comp(style_file)
        style_file = style_file.to(self.device)
        return style_file

    def painterly_rendering(self, prompt, style_fpath):
        # load style file
        style_path = Path(style_fpath)
        assert style_path.exists(), f"{style_fpath} is not exist!"
        self.print(f"load style file from: {style_path.as_posix()}")
        style_pil = Image.open(style_path.as_posix()).convert("RGB")
        style_img = self.style_file_preprocess(style_pil)
        shutil.copy(style_fpath, self.result_path)  # copy style file

        # extract style features from style image
        feat_style = None
        for i in range(5):
            with torch.no_grad():
                # r is region of interest (mask)
                feat_e = self.style_extractor.forward_samples_hypercolumn(style_img, samps=1000)
                feat_style = feat_e if feat_style is None else torch.cat((feat_style, feat_e), dim=2)

        # text prompt encoding
        self.print(f"prompt: {prompt}")
        text_tokenize = self.tokenize_fn(prompt).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(text_tokenize)

        renderer = Painter(self.x_cfg,
                           self.args.diffvg,
                           num_strokes=self.x_cfg.num_paths,
                           canvas_size=self.x_cfg.image_size,
                           device=self.device)
        img = renderer.init_image(stage=0)
        self.print("init_image shape: ", img.shape)
        plot_img(img, self.result_path, fname="init_img")

        optimizer = PainterOptimizer(renderer, self.x_cfg.lr, self.x_cfg.width_lr, self.x_cfg.color_lr)
        optimizer.init_optimizers()

        style_weight = 4 * (self.x_cfg.style_strength / 100)
        self.print(f'style_weight: {style_weight}')

        total_step = self.x_cfg.num_iter
        with tqdm(initial=self.step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_step:
                rendering = renderer.get_image(self.step).to(self.device)

                if self.make_video and (self.step % self.args.framefreq == 0 or self.step == total_step - 1):
                    plot_img(rendering, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                rendering_aug = self.drawing_augment(rendering)

                loss = torch.tensor(0., device=self.device)

                # do clip optimization
                if self.step < 0.9 * total_step:
                    for n in range(self.x_cfg.num_aug):
                        loss -= torch.cosine_similarity(text_features, rendering_aug[n:n + 1], dim=1).mean()

                # do style optimization
                # extract style features based on the approach from STROTSS [Kolkin et al., 2019].
                feat_content = self.style_extractor(rendering)

                xx, xy = sample_indices(feat_content[0], feat_style)

                np.random.shuffle(xx)
                np.random.shuffle(xy)

                L_style = self.style_loss.forward(feat_content, feat_content, feat_style, [xx, xy], 0)

                loss += L_style * style_weight

                pbar.set_description(
                    f"lr: {optimizer.get_lr():.3f}, "
                    f"L_train: {loss.item():.4f}, "
                    f"L_style: {L_style.item():.4f}"
                )

                # optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                renderer.clip_curve_shape()

                if self.x_cfg.lr_schedule:
                    optimizer.update_lr(self.step)

                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    plot_couple(style_img,
                                rendering,
                                self.step,
                                prompt=prompt,
                                output_dir=self.png_logs_dir.as_posix(),
                                fname=f"iter{self.step}")
                    renderer.save_svg(self.svg_logs_dir.as_posix(), f"svg_iter{self.step}")

                self.step += 1
                pbar.update(1)

        plot_couple(style_img,
                    rendering,
                    self.step,
                    prompt=prompt,
                    output_dir=self.result_path.as_posix(),
                    fname=f"final_iter")
        renderer.save_svg(self.result_path.as_posix(), "final_render")

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.result_path / "styleclipdraw_rendering.mp4").as_posix()
            ])

        self.close(msg="painterly rendering complete.")
