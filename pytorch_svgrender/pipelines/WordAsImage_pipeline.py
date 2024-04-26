# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
from pathlib import Path

from tqdm.auto import tqdm
import torch

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.painter.wordasimage import Painter, PainterOptimizer
from pytorch_svgrender.painter.wordasimage.losses import ToneLoss, ConformalLoss
from pytorch_svgrender.painter.vectorfusion import LSDSPipeline
from pytorch_svgrender.plt import plot_img, plot_couple
from pytorch_svgrender.diffusers_warp import init_StableDiffusion_pipeline
from pytorch_svgrender.svgtools import FONT_LIST


class WordAsImagePipeline(ModelState):

    def __init__(self, args):
        # assert
        assert args.x.optim_letter in args.x.word
        assert Path(args.x.font_path).exists(), f"{args.x.font_path} is not exist."
        assert args.x.font in FONT_LIST, f"{args.x.font} is not currently supported."

        # make logdir
        logdir_ = f"sd{args.seed}" \
                  f"-im{args.x.image_size}" \
                  f"-{args.x.word}-{args.x.optim_letter}"
        super().__init__(args, log_path_suffix=logdir_)

        # log dir
        self.png_log_dir = self.result_path / "png_logs"
        self.svg_log_dir = self.result_path / "svg_logs"
        # font
        self.font = self.x_cfg.font
        self.font_path = self.x_cfg.font_path
        self.optim_letter = self.x_cfg.optim_letter
        # letter
        self.letter = self.x_cfg.optim_letter
        self.target_letter = self.result_path / f"{self.font}_{self.optim_letter}_scaled.svg"
        # make log dir
        if self.accelerator.is_main_process:
            self.png_log_dir.mkdir(parents=True, exist_ok=True)
            self.svg_log_dir.mkdir(parents=True, exist_ok=True)

        # make video log
        self.make_video = self.args.mv
        if self.make_video:
            self.frame_idx = 0
            self.frame_log_dir = self.result_path / "frame_logs"
            self.frame_log_dir.mkdir(parents=True, exist_ok=True)

        self.diffusion = init_StableDiffusion_pipeline(
            self.x_cfg.model_id,
            custom_pipeline=LSDSPipeline,
            device=self.device,
            local_files_only=not args.diffuser.download,
            force_download=args.diffuser.force_download,
            resume_download=args.diffuser.resume_download,
            ldm_speed_up=self.x_cfg.ldm_speed_up,
            enable_xformers=self.x_cfg.enable_xformers,
            gradient_checkpoint=self.x_cfg.gradient_checkpoint,
            lora_path=self.x_cfg.lora_path
        )

        self.g_device = torch.Generator(device=self.device).manual_seed(args.seed)

    def painterly_rendering(self, word, semantic_concept, optimized_letter):
        prompt = semantic_concept + ". " + self.x_cfg.prompt_suffix
        self.print(f"prompt: {prompt}")

        # load the optimized letter
        renderer = Painter(self.font, canvas_size=self.x_cfg.image_size, device=self.device)

        # font to svg
        self.print(f"font type: {self.font}\n")
        renderer.preprocess_font(word,
                                 optimized_letter,
                                 self.x_cfg.level_of_cc,
                                 self.font_path,
                                 self.result_path.as_posix())

        # init letter shape
        img_init = renderer.init_shape(self.target_letter)
        plot_img(img_init, self.result_path, fname="word_init")

        # save init letter
        renderer.pretty_save_svg(self.result_path / "letter_init.svg")
        init_letter = renderer.get_image()

        n_iter = self.x_cfg.num_iter

        # init optimizer and lr_schedular
        optimizer = PainterOptimizer(renderer, n_iter, self.x_cfg.lr)
        optimizer.init_optimizers()

        # init Tone loss
        if self.x_cfg.tone_loss.use:
            tone_loss = ToneLoss(self.x_cfg.tone_loss)
            tone_loss.set_image_init(img_init)

        # init conformal loss
        if self.x_cfg.conformal.use:
            conformal_loss = ConformalLoss(renderer.get_point_parameters(),
                                           renderer.shape_groups,
                                           optimized_letter, self.device)

        with tqdm(initial=self.step, total=n_iter, disable=not self.accelerator.is_main_process) as pbar:
            for i in range(n_iter):

                raster_img = renderer.get_image(step=i)

                if self.make_video and (i % self.args.framefreq == 0 or i == n_iter - 1):
                    plot_img(raster_img, self.frame_log_dir, fname=f"iter{self.step}")

                L_sds, grad = self.diffusion.score_distillation_sampling(
                    raster_img,
                    im_size=self.x_cfg.sds.im_size,
                    prompt=[prompt],
                    negative_prompt=[self.args.neg_prompt],
                    guidance_scale=self.x_cfg.sds.guidance_scale,
                    grad_scale=self.x_cfg.sds.grad_scale,
                    t_range=list(self.x_cfg.sds.t_range),
                )

                loss = L_sds

                if self.x_cfg.tone_loss.use:
                    tone_loss_res = tone_loss(raster_img, step=i)
                    loss = loss + tone_loss_res

                if self.x_cfg.conformal.use:
                    loss_angles = conformal_loss()
                    loss_angles = self.x_cfg.conformal.angeles_w * loss_angles
                    loss = loss + loss_angles

                pbar.set_description(
                    f"n_params: {len(renderer.get_point_parameters())}, "
                    f"lr: {optimizer.get_lr():.4f}, "
                    f"L_total: {loss.item():.4f}, "
                )

                # optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                if self.x_cfg.lr_schedule:
                    optimizer.update_lr()

                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    plot_couple(init_letter,
                                raster_img,
                                self.step,
                                output_dir=self.png_log_dir.as_posix(),
                                fname=f"iter{self.step}",
                                prompt=prompt)
                    renderer.pretty_save_svg(self.svg_log_dir / f"svg_iter{self.step}.svg")

                self.step += 1
                pbar.update(1)

        # save final optimized letter
        renderer.pretty_save_svg(self.result_path / "final_letter.svg")

        # combine word
        renderer.combine_word(word, optimized_letter, self.font, self.result_path)

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.result_path / "wordasimg_rendering.mp4").as_posix()
            ])

        self.close(msg="painterly rendering complete.")
