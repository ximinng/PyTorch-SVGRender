# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import torch
from tqdm.auto import tqdm
from torchvision import transforms
import clip

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.painter.clipdraw import Painter, PainterOptimizer
from pytorch_svgrender.plt import plot_img, plot_couple


class CLIPDrawPipeline(ModelState):

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

        self.clip, self.tokenize_fn = self.init_clip()

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

    def painterly_rendering(self, prompt):
        self.print(f"prompt: {prompt}")

        # text prompt encoding
        text_tokenize = self.tokenize_fn(prompt).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(text_tokenize)

        # init SVG Painter
        renderer = Painter(self.x_cfg,
                           self.args.diffvg,
                           num_strokes=self.x_cfg.num_paths,
                           canvas_size=self.x_cfg.image_size,
                           device=self.device)
        img = renderer.init_image(stage=0)
        self.print("init_image shape: ", img.shape)
        plot_img(img, self.result_path, fname="init_img")

        # init painter optimizer
        optimizer = PainterOptimizer(renderer, self.x_cfg.lr, self.x_cfg.width_lr, self.x_cfg.color_lr)
        optimizer.init_optimizers()

        total_step = self.x_cfg.num_iter
        with tqdm(initial=self.step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_step:
                rendering = renderer.get_image(self.step).to(self.device)

                if self.make_video and (self.step % self.args.framefreq == 0 or self.step == total_step - 1):
                    plot_img(rendering, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                # data augmentation
                aug_svg_batch = self.drawing_augment(rendering)

                loss = torch.tensor(0., device=self.device)
                for n in range(self.x_cfg.num_aug):
                    loss -= torch.cosine_similarity(text_features, aug_svg_batch[n:n + 1], dim=1).mean()

                pbar.set_description(
                    f"lr: {optimizer.get_lr():.3f}, "
                    f"L_train: {loss.item():.4f}"
                )

                # optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                renderer.clip_curve_shape()

                if self.x_cfg.lr_schedule:
                    optimizer.update_lr(self.step)

                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    plot_couple(img,
                                rendering,
                                self.step,
                                prompt=prompt,
                                output_dir=self.png_logs_dir.as_posix(),
                                fname=f"iter{self.step}")
                    renderer.save_svg(self.svg_logs_dir.as_posix(), f"svg_iter{self.step}")

                self.step += 1
                pbar.update(1)

        renderer.save_svg(self.result_path.as_posix(), "final_render")

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.result_path / "clipdraw_rendering.mp4").as_posix()
            ])

        self.close(msg="painterly rendering complete.")
