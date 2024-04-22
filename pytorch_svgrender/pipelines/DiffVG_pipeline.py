# -*- coding: utf-8 -*-
# Author: ximing
# Description: DiffVG pipeline
# Copyright (c) 2023, XiMing Xing.
# License: MIT License

import shutil
from pathlib import Path
from functools import partial
from typing import AnyStr
from PIL import Image

from tqdm.auto import tqdm
import torch
from torchvision import transforms

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.libs.metric.lpips_origin import LPIPS
from pytorch_svgrender.painter.diffvg import Painter, PainterOptimizer
from pytorch_svgrender.plt import plot_img, plot_couple


class DiffVGPipeline(ModelState):

    def __init__(self, args):
        logdir_ = f"sd{args.seed}" \
                  f"-{args.x.path_type}" \
                  f"-P{args.x.num_paths}"
        super().__init__(args, log_path_suffix=logdir_)

        assert self.x_cfg.path_type in ['unclosed', 'closed']

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

    def target_file_preprocess(self, tar_path):
        process_comp = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ])

        tar_pil = Image.open(tar_path).convert("RGB")  # open file
        target_img = process_comp(tar_pil)  # preprocess
        target_img = target_img.to(self.device)
        return target_img

    def painterly_rendering(self, img_path: AnyStr):
        # load target file
        target_file = Path(img_path)
        assert target_file.exists(), f"{target_file} is not exist!"
        shutil.copy(target_file, self.result_path)  # copy target file
        target_img = self.target_file_preprocess(target_file.as_posix())
        self.print(f"load image from: '{target_file.as_posix()}'")

        # init Painter
        renderer = Painter(target_img,
                           self.args.diffvg,
                           canvas_size=[target_img.shape[3], target_img.shape[2]],
                           path_type=self.x_cfg.path_type,
                           max_width=self.x_cfg.max_width,
                           device=self.device)
        init_img = renderer.init_image(num_paths=self.x_cfg.num_paths)
        self.print("init_image shape: ", init_img.shape)
        plot_img(init_img, self.result_path, fname="init_img")

        # init Painter Optimizer
        num_iter = self.x_cfg.num_iter
        optimizer = PainterOptimizer(renderer,
                                     num_iter,
                                     self.x_cfg.lr_base,
                                     trainable_stroke=self.x_cfg.path_type == 'unclosed')
        optimizer.init_optimizer()

        # Set Loss
        if self.x_cfg.loss_type in ['lpips', 'l2+lpips']:
            lpips_loss_fn = LPIPS(net=self.x_cfg.perceptual.lpips_net).to(self.device)
            perceptual_loss_fn = partial(lpips_loss_fn.forward, return_per_layer=False, normalize=False)

        with tqdm(initial=self.step, total=num_iter, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < num_iter:
                raster_img = renderer.get_image(self.step).to(self.device)

                if self.make_video and (self.step % self.args.framefreq == 0 or self.step == num_iter - 1):
                    plot_img(raster_img, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                # Reconstruction Loss
                if self.x_cfg.loss_type == 'l1':
                    loss_recon = torch.nn.functional.l1_loss(raster_img, target_img)
                elif self.x_cfg.loss_type == 'lpips':
                    loss_recon = perceptual_loss_fn(raster_img, target_img).mean()
                elif self.x_cfg.loss_type == 'l2':  # default: MSE loss
                    loss_recon = torch.nn.functional.mse_loss(raster_img, target_img)
                elif self.x_cfg.loss_type == 'l2+lpips':  # default: MSE loss
                    lpips = perceptual_loss_fn(raster_img, target_img).mean()
                    loss_mse = torch.nn.functional.mse_loss(raster_img, target_img)
                    loss_recon = loss_mse + lpips

                # total loss
                loss = loss_recon

                pbar.set_description(
                    f"lr: {optimizer.get_lr():.4f}, "
                    f"L_recon: {loss_recon.item():.4f}"
                )

                # optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                renderer.clip_curve_shape()

                if self.x_cfg.lr_schedule:
                    optimizer.update_lr()

                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    plot_couple(target_img,
                                raster_img,
                                self.step,
                                output_dir=self.png_logs_dir.as_posix(),
                                fname=f"iter{self.step}")
                    renderer.save_svg(self.svg_logs_dir / f"svg_iter{self.step}.svg")

                self.step += 1
                pbar.update(1)

        # end rendering
        renderer.save_svg(self.result_path / "final_render.svg")

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.result_path / "diffvg_rendering.mp4").as_posix()
            ])

        self.close(msg="painterly rendering complete.")
