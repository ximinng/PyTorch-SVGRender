# -*- coding: utf-8 -*-
# Author: ximing
# Description: LIVE pipeline
# Copyright (c) 2023, XiMing Xing.
# License: MIT License

import shutil
from pathlib import Path
from typing import AnyStr
from PIL import Image

from tqdm.auto import tqdm
import torch
from torchvision import transforms

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.painter.live import Painter, PainterOptimizer, xing_loss_fn
from pytorch_svgrender.plt import plot_img, plot_couple


class LIVEPipeline(ModelState):

    def __init__(self, args):
        logdir_ = f"sd{args.seed}" \
                  f"-im{args.x.image_size}" \
                  f"-P{args.x.num_paths}"
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

    def get_path_schedule(self, schedule_each):
        if self.x_cfg.path_schedule == 'repeat':
            return int(self.x_cfg.num_paths / schedule_each) * [schedule_each]
        elif self.x_cfg.path_schedule == 'list':
            assert isinstance(self.x_cfg.schedule_each, list)
            return schedule_each
        else:
            raise NotImplementedError

    def target_file_preprocess(self, tar_path):
        process_comp = transforms.Compose([
            transforms.Resize(size=(self.x_cfg.image_size, self.x_cfg.image_size)),
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
        self.print(f"load image file from: '{target_file.as_posix()}'")

        # log path_schedule
        path_schedule = self.get_path_schedule(self.x_cfg.schedule_each)
        self.print(f"path_schedule: {path_schedule}")

        renderer = Painter(target_img,
                           self.args.diffvg,
                           self.x_cfg.num_segments,
                           self.x_cfg.segment_init,
                           self.x_cfg.radius,
                           canvas_size=self.x_cfg.image_size,
                           trainable_bg=self.x_cfg.trainable_bg,
                           stroke=self.x_cfg.train_stroke,
                           stroke_width=self.x_cfg.width,
                           device=self.device)
        # first init center
        renderer.component_wise_path_init(pred=None, init_type=self.x_cfg.coord_init)

        num_iter = self.x_cfg.num_iter

        optimizer_list = [
            PainterOptimizer(renderer, num_iter, self.x_cfg.lr_base,
                             self.x_cfg.train_stroke, self.x_cfg.trainable_bg)
            for _ in range(len(path_schedule))
        ]

        pathn_record = []
        loss_weight_keep = 0
        loss_weight = 1

        total_step = len(path_schedule) * num_iter
        with tqdm(initial=self.step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:

            for path_idx, pathn in enumerate(path_schedule):
                # record path
                pathn_record.append(pathn)
                # init graphic
                img = renderer.init_image(num_paths=pathn)
                plot_img(img, self.result_path, fname=f"init_img_{path_idx}")
                # rebuild optimizer
                optimizer_list[path_idx].init_optimizers()

                pbar.write(f"=> adding {pathn} paths, n_path: {sum(pathn_record)}, "
                           f"path_schedule: {self.x_cfg.path_schedule}")

                for t in range(num_iter):
                    raster_img = renderer.get_image(step=t).to(self.device)

                    if self.make_video and (t % self.args.framefreq == 0 or t == num_iter - 1):
                        plot_img(raster_img, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                        self.frame_idx += 1

                    if self.x_cfg.use_distance_weighted_loss:
                        loss_weight = renderer.calc_distance_weight(loss_weight_keep)

                    # UDF Loss for Reconstruction
                    if self.x_cfg.use_l1_loss:
                        loss_recon = torch.nn.functional.l1_loss(raster_img, target_img)
                    else:  # default: MSE loss
                        loss_mse = ((raster_img - target_img) ** 2)
                        loss_recon = (loss_mse.sum(1) * loss_weight).mean()

                    # Xing Loss for Self-Interaction Problem
                    loss_xing = xing_loss_fn(renderer.get_point_parameters()) * self.x_cfg.xing_loss_weight
                    # total loss
                    loss = loss_recon + loss_xing

                    pbar.set_description(
                        f"lr: {optimizer_list[path_idx].get_lr():.4f}, "
                        f"L_total: {loss.item():.4f}, "
                        f"L_recon: {loss_recon.item():.4f}, "
                        f"L_xing: {loss_xing.item()}"
                    )

                    # optimization
                    for i in range(path_idx + 1):
                        optimizer_list[i].zero_grad_()

                    loss.backward()

                    for i in range(path_idx + 1):
                        optimizer_list[i].step_()

                    renderer.clip_curve_shape()

                    if self.x_cfg.lr_schedule:
                        for i in range(path_idx + 1):
                            optimizer_list[i].update_lr()

                    if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                        plot_couple(target_img,
                                    raster_img,
                                    self.step,
                                    output_dir=self.png_logs_dir.as_posix(),
                                    fname=f"iter{self.step}")
                        renderer.save_svg(self.svg_logs_dir / f"svg_iter{self.step}.svg")

                    self.step += 1
                    pbar.update(1)

                # end a set of path optimization
                if self.x_cfg.use_distance_weighted_loss:
                    loss_weight_keep = loss_weight.detach().cpu().numpy() * 1
                # recalculate the coordinates for the new join path
                renderer.component_wise_path_init(pred=raster_img, init_type=self.x_cfg.coord_init)

        renderer.save_svg(self.result_path / "final_svg.svg")

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.result_path / "live_rendering.mp4").as_posix()
            ])

        self.close(msg="painterly rendering complete.")
