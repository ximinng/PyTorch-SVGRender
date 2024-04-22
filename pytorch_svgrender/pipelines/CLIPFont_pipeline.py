# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
from PIL import Image
from typing import AnyStr
import pathlib

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm
from svgutils.transform import fromfile

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.plt import plot_img, plot_couple, plot_img_title
from pytorch_svgrender.painter.clipfont import (imagenet_templates, compose_text_with_templates, Painter,
                                                PainterOptimizer)
from pytorch_svgrender.libs.metric.clip_score import CLIPScoreWrapper
from pytorch_svgrender.libs.metric.piq.perceptual import LPIPS


class CLIPFontPipeline(ModelState):

    def __init__(self, args):
        logdir_ = f"sd{args.seed}" \
                  f"-lpips{args.x.lam_lpips}-l2{args.x.lam_l2}" \
                  f"{f'-{args.x.font.reinit_color}' if args.x.font.reinit else ''}"
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

        # init clip model
        self.clip_wrapper = CLIPScoreWrapper(self.x_cfg.clip.model_name, device=self.device)
        # init LPIPS
        self.lam_lpips = 0 if self.x_cfg.get('lam_lpips', None) is None else self.x_cfg.lam_lpips
        self.lpips_fn = LPIPS()
        # l2
        self.lam_l2 = 0 if self.x_cfg.get('lam_l2', None) is None else self.x_cfg.lam_l2

    def load_target_file(self, tar_path: AnyStr, image_size: int = 224):
        process_comp = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ])

        tar_pil = Image.open(tar_path).convert("RGB")  # open file
        target_img = process_comp(tar_pil)  # preprocess
        return target_img.to(self.device)

    def cropper(self, x: torch.Tensor) -> torch.Tensor:
        return transforms.RandomCrop(self.x_cfg.crop_size)(x)

    def padding_cropper(self, x: torch.Tensor) -> torch.Tensor:
        return transforms.RandomCrop(size=500, padding=100, fill=255, padding_mode='constant')(x)

    def affine_to512(self, x: torch.Tensor) -> torch.Tensor:
        comp = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.3),
            transforms.Resize(512)
        ])
        return comp(x)

    def resize224_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, size=224, mode='bicubic')
        return self.clip_wrapper.norm_(x)

    def painterly_rendering(self, svg_path, prompt):
        svg_path = pathlib.Path(svg_path)
        assert svg_path.exists(), f"'{svg_path}' is not exist."

        # load renderer
        renderer = self.load_renderer()

        # rescale svg
        fig = fromfile(svg_path.as_posix())
        fig.set_size(('512', '512'))
        filename = str(svg_path.name).split('.')[0]
        svg_path = self.result_path / f'{filename}_scale.svg'
        fig.save(svg_path.as_posix())

        # init shapes and shape groups
        init_img = renderer.init_shapes(svg_path.as_posix(), reinit_cfg=self.x_cfg.font)
        self.print("init_image shape: ", init_img.shape)
        plot_img(init_img, self.result_path, fname="init_img")

        # load init file
        with torch.no_grad():
            source_image = self.load_target_file(self.result_path / 'init_img.png', image_size=512)
            source_image = source_image.detach()
            source_image_feats = self.clip_wrapper.encode_image(self.resize224_norm(source_image)).detach()

        # build optimizer
        optimizer = PainterOptimizer(renderer, self.x_cfg.lr_base)
        optimizer.init_optimizers()

        # pre-calc
        with torch.no_grad():
            # encode text prompt and source prompt
            template_text = compose_text_with_templates(prompt, imagenet_templates)
            text_features = self.clip_wrapper.encode_text(template_text).detach()
            source = "A photo"
            template_source = compose_text_with_templates(source, imagenet_templates)
            text_source = self.clip_wrapper.encode_text(template_source).detach()

        total_step = self.x_cfg.num_iter
        with tqdm(initial=self.step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_step:
                img_t = renderer.get_image().to(self.device)

                if self.make_video and (self.step % self.args.framefreq == 0 or self.step == total_step - 1):
                    plot_img(img_t, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                # style loss
                # directional loss 1
                img_proc = []
                for n in range(self.x_cfg.num_crops):
                    target_crop = self.cropper(img_t)
                    target_crop = self.affine_to512(target_crop)
                    img_proc.append(target_crop)
                img_aug = torch.cat(img_proc, dim=0)
                image_features = self.clip_wrapper.encode_image(self.resize224_norm(img_aug))

                loss_patch = self.x_cfg.lam_patch * self.clip_wrapper.directional_loss(text_source,
                                                                                       source_image_feats,
                                                                                       text_features,
                                                                                       image_features,
                                                                                       self.x_cfg.thresh)

                # directional loss 2
                img_proc2 = []
                for n in range(32):
                    target_crop = self.padding_cropper(img_t)
                    target_crop = self.affine_to512(target_crop)
                    img_proc2.append(target_crop)
                img_aug2 = torch.cat(img_proc2, dim=0)
                glob_features = self.clip_wrapper.encode_image(self.resize224_norm(img_aug2))

                loss_glob = self.x_cfg.lam_dir * self.clip_wrapper.directional_loss(text_source,
                                                                                    source_image_feats,
                                                                                    text_features, glob_features)

                # LPIPS
                loss_lpips = self.lam_lpips * self.lpips_fn(img_t, source_image)

                # L2
                loss_l2 = self.lam_l2 * F.mse_loss(img_t, source_image)

                # total loss
                loss = loss_patch + loss_glob + loss_lpips + loss_l2

                # log
                p_lr, c_lr = optimizer.get_lr()
                pbar.set_description(
                    f"point_lr: {p_lr}, color_lr: {c_lr}, "
                    f"L_total: {loss.item():.4f}, "
                    f"L_patch: {loss_patch.item():.4f}, "
                    f"L_glob: {loss_glob.item():.4f}, "
                    f"L_lpips: {loss_lpips.item():.4f}, "
                    f"L_l2: {loss_l2.item():.4f}."
                )

                # backward and optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                renderer.clip_curve_shape()

                if self.x_cfg.lr_schedule:
                    optimizer.update_lr(self.step)

                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    plot_couple(init_img,
                                img_t,
                                self.step,
                                output_dir=self.png_logs_dir.as_posix(),
                                fname=f"iter{self.step}")
                    renderer.pretty_save_svg(self.svg_logs_dir / f"svg_iter{self.step}.svg")

                self.step += 1
                pbar.update(1)

        # log final results
        renderer.pretty_save_svg(self.result_path / "final_render.svg")
        final_raster_sketch = renderer.get_image().to(self.device)
        plot_img_title(final_raster_sketch,
                       title=f'final result - {self.step} step',
                       output_dir=self.result_path,
                       fname='final_render')

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.result_path / "clipfont_rendering.mp4").as_posix()
            ])

        self.close(msg="painterly rendering complete.")

    def load_renderer(self):
        renderer = Painter(device=self.device)
        return renderer
