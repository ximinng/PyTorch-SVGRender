# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

from PIL import Image
from typing import Union, AnyStr, List

from omegaconf.listconfig import ListConfig
import diffusers
import numpy as np
from tqdm.auto import tqdm
import torch
from torchvision import transforms
import clip

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.painter.vectorfusion import LSDSPipeline, LSDSSDXLPipeline, Painter, PainterOptimizer
from pytorch_svgrender.painter.vectorfusion import channel_saturation_penalty_loss as pixel_penalty_loss
from pytorch_svgrender.painter.live import xing_loss_fn
from pytorch_svgrender.plt import plot_img, plot_couple
from pytorch_svgrender.token2attn.ptp_utils import view_images
from pytorch_svgrender.diffusers_warp import init_StableDiffusion_pipeline, model2res


class VectorFusionPipeline(ModelState):

    def __init__(self, args):
        assert args.x.style in ["iconography", "pixelart", "low-poly", "painting", "sketch", "ink"]

        logdir_ = f"sd{args.seed}-" \
                  f"{'scratch' if args.x.skip_live else 'baseline'}" \
                  f"-{args.x.model_id}" \
                  f"-{args.x.style}" \
                  f"-P{args.x.num_paths}" \
                  f"{'-RePath' if args.x.path_reinit.use else ''}"
        super().__init__(args, log_path_suffix=logdir_)

        # create log dir
        self.png_logs_dir = self.result_path / "png_logs"
        self.svg_logs_dir = self.result_path / "svg_logs"
        self.ft_png_logs_dir = self.result_path / "ft_png_logs"
        self.ft_svg_logs_dir = self.result_path / "ft_svg_logs"
        self.sd_sample_dir = self.result_path / 'sd_samples'
        self.reinit_dir = self.result_path / "reinit_logs"

        if self.accelerator.is_main_process:
            self.png_logs_dir.mkdir(parents=True, exist_ok=True)
            self.svg_logs_dir.mkdir(parents=True, exist_ok=True)
            self.ft_png_logs_dir.mkdir(parents=True, exist_ok=True)
            self.ft_svg_logs_dir.mkdir(parents=True, exist_ok=True)
            self.sd_sample_dir.mkdir(parents=True, exist_ok=True)
            self.reinit_dir.mkdir(parents=True, exist_ok=True)

        self.select_fpth = self.result_path / 'select_sample.png'

        # make video log
        self.make_video = self.args.mv
        if self.make_video:
            self.frame_idx = 0
            self.frame_log_dir = self.result_path / "frame_logs"
            self.frame_log_dir.mkdir(parents=True, exist_ok=True)

        if self.x_cfg.model_id == "sdxl":
            # default LSDSSDXLPipeline scheduler is EulerDiscreteScheduler
            # when LSDSSDXLPipeline calls, scheduler.timesteps will change in step 4
            # which causes problem in sds add_noise() function
            # because the random t may not in scheduler.timesteps
            custom_pipeline = LSDSSDXLPipeline
            custom_scheduler = diffusers.DPMSolverMultistepScheduler
        elif self.x_cfg.model_id == 'sd21':
            custom_pipeline = LSDSPipeline
            custom_scheduler = diffusers.DDIMScheduler
        else:  # sd14, sd15
            custom_pipeline = LSDSPipeline
            custom_scheduler = diffusers.PNDMScheduler

        self.diffusion = init_StableDiffusion_pipeline(
            self.x_cfg.model_id,
            custom_pipeline=custom_pipeline,
            custom_scheduler=custom_scheduler,
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

        self.style = self.x_cfg.style
        if self.style in ["pixelart", "low-poly"]:
            self.x_cfg.path_schedule = 'list'
            self.x_cfg.schedule_each = list([args.x.grid])

        if self.style == "pixelart":
            self.x_cfg.lr_stage_one.lr_schedule = False
            self.x_cfg.lr_stage_two.lr_schedule = False

    def get_path_schedule(self, schedule_each: Union[int, List]):
        if self.x_cfg.path_schedule == 'repeat':
            return int(self.x_cfg.num_paths / schedule_each) * [schedule_each]
        elif self.x_cfg.path_schedule == 'list':
            assert isinstance(self.x_cfg.schedule_each, list) or \
                   isinstance(self.x_cfg.schedule_each, ListConfig)
            return schedule_each
        else:
            raise NotImplementedError

    def target_file_preprocess(self, tar_path: AnyStr):
        process_comp = transforms.Compose([
            transforms.Resize(size=(self.x_cfg.image_size, self.x_cfg.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ])

        tar_pil = Image.open(tar_path).convert("RGB")  # open file
        target_img = process_comp(tar_pil)  # preprocess
        target_img = target_img.to(self.device)
        return target_img

    @torch.no_grad()
    def rejection_sampling(self, img_caption: Union[AnyStr, List], diffusion_samples: List):
        clip_model, preprocess = clip.load("ViT-B/32", device=self.device)

        text_input = clip.tokenize([img_caption]).to(self.device)
        text_features = clip_model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        clip_images = torch.stack([
            preprocess(sample) for sample in diffusion_samples]
        ).to(self.device)
        image_features = clip_model.encode_image(clip_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # clip score
        similarity_scores = (text_features @ image_features.T).squeeze(0)

        selected_image_index = similarity_scores.argmax().item()
        selected_image = diffusion_samples[selected_image_index]
        return selected_image

    def diffusion_sampling(self, text_prompt: AnyStr):
        """sampling K images"""
        diffusion_samples = []
        for i in range(self.x_cfg.K):
            height = width = model2res(self.x_cfg.model_id)
            outputs = self.diffusion(prompt=[text_prompt],
                                     negative_prompt=[self.args.neg_prompt],
                                     height=height,
                                     width=width,
                                     num_images_per_prompt=1,
                                     num_inference_steps=self.x_cfg.num_inference_steps,
                                     guidance_scale=self.x_cfg.guidance_scale,
                                     generator=self.g_device)
            outputs_np = [np.array(img) for img in outputs.images]
            view_images(outputs_np, save_image=True, fp=self.sd_sample_dir / f'samples_{i}.png')
            diffusion_samples.extend(outputs.images)

        self.print(f"num_generated_samples: {len(diffusion_samples)}, shape: {outputs_np[0].shape}")

        return diffusion_samples

    def LIVE_rendering(self, text_prompt: AnyStr):
        select_fpth = self.select_fpth
        # sampling K images
        diffusion_samples = self.diffusion_sampling(text_prompt)
        # rejection sampling
        select_target = self.rejection_sampling(text_prompt, diffusion_samples)
        select_target_pil = Image.fromarray(np.asarray(select_target))  # numpy to PIL
        select_target_pil.save(select_fpth)

        # load target file
        assert select_fpth.exists(), f"{select_fpth} is not exist!"
        target_img = self.target_file_preprocess(select_fpth.as_posix())
        self.print(f"load target file from: {select_fpth.as_posix()}")

        # log path_schedule
        path_schedule = self.get_path_schedule(self.x_cfg.schedule_each)
        self.print(f"path_schedule: {path_schedule}")

        renderer = self.load_renderer()
        # first init center
        renderer.component_wise_path_init(target_img, pred=None, init_type=self.x_cfg.coord_init)

        optimizer_list = [PainterOptimizer(renderer, self.style, self.x_cfg.num_iter,
                                           self.x_cfg.lr_stage_one, self.x_cfg.trainable_bg)
                          for _ in range(len(path_schedule))]

        pathn_record = []
        loss_weight_keep = 0

        total_step = len(path_schedule) * self.x_cfg.num_iter
        with tqdm(initial=self.step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            for path_idx, pathn in enumerate(path_schedule):
                # record path
                pathn_record.append(pathn)
                # init graphic
                img = renderer.init_image(stage=0, num_paths=pathn)
                plot_img(img, self.result_path, fname=f"init_img_{path_idx}")
                # rebuild optimizer
                optimizer_list[path_idx].init_optimizers(pid_delta=int(path_idx * pathn))

                pbar.write(f"=> adding {pathn} paths, n_path: {sum(pathn_record)}, "
                           f"n_points: {len(renderer.get_point_parameters())}, "
                           f"n_colors: {len(renderer.get_color_parameters())}")

                for t in range(self.x_cfg.num_iter):
                    raster_img = renderer.get_image(step=t).to(self.device)

                    if self.make_video and (self.step % self.args.framefreq == 0 or self.step == total_step - 1):
                        plot_img(raster_img, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                        self.frame_idx += 1

                    if self.x_cfg.use_distance_weighted_loss and not (self.style == "pixelart"):
                        loss_weight = renderer.calc_distance_weight(loss_weight_keep)

                    # reconstruction loss
                    if self.style == "pixelart":
                        loss_recon = torch.nn.functional.l1_loss(raster_img, target_img)
                    else:  # UDF loss
                        loss_recon = ((raster_img - target_img) ** 2)
                        loss_recon = (loss_recon.sum(1) * loss_weight).mean()

                    # Xing Loss for Self-Interaction Problem
                    loss_xing = torch.tensor(0.)
                    if self.style == "iconography":
                        loss_xing = xing_loss_fn(renderer.get_point_parameters()) * self.x_cfg.xing_loss_weight

                    # total loss
                    loss = loss_recon + loss_xing

                    lr_str = ""
                    for k, lr in optimizer_list[path_idx].get_lr().items():
                        lr_str += f"{k}_lr: {lr:.4f}, "

                    pbar.set_description(
                        lr_str +
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

                    if self.x_cfg.lr_stage_one.lr_schedule:
                        for i in range(path_idx + 1):
                            optimizer_list[i].update_lr()

                    if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                        plot_couple(target_img,
                                    raster_img,
                                    self.step,
                                    prompt=text_prompt,
                                    output_dir=self.png_logs_dir.as_posix(),
                                    fname=f"iter{self.step}")
                        renderer.pretty_save_svg(self.svg_logs_dir / f"svg_iter{self.step}.svg")

                    self.step += 1
                    pbar.update(1)

                # end a set of path optimization
                if self.x_cfg.use_distance_weighted_loss and not (self.style == "pixelart"):
                    loss_weight_keep = loss_weight.detach().cpu().numpy() * 1
                # recalculate the coordinates for the new join path
                renderer.component_wise_path_init(target_img, raster_img)

        # end LIVE
        final_svg_fpth = self.result_path / "live_stage_one_final.svg"
        renderer.pretty_save_svg(final_svg_fpth)

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.result_path / "VF_rendering_stage1.mp4").as_posix()
            ])

        return target_img, final_svg_fpth

    def painterly_rendering(self, text_prompt: AnyStr):
        # log prompts
        self.print(f"prompt: {text_prompt}")
        self.print(f"negative_prompt: {self.args.neg_prompt}\n")

        if self.x_cfg.skip_live:
            target_img = torch.randn(1, 3, self.x_cfg.image_size, self.x_cfg.image_size)
            final_svg_fpth = None
            self.print("from scratch with Score Distillation Sampling...")
        else:
            # text-to-img-to-svg
            target_img, final_svg_fpth = self.LIVE_rendering(text_prompt)
            torch.cuda.empty_cache()
            self.x_cfg.path_svg = final_svg_fpth
            self.print("\nfine-tune SVG via Score Distillation Sampling...")

        renderer = self.load_renderer(path_svg=final_svg_fpth)

        if self.x_cfg.skip_live:
            renderer.component_wise_path_init(target_img, pred=None, init_type='random')

        img = renderer.init_image(stage=0, num_paths=self.x_cfg.num_paths)
        plot_img(img, self.result_path, fname=f"init_img_stage_two")

        optimizer = PainterOptimizer(renderer, self.style,
                                     self.x_cfg.sds.num_iter,
                                     self.x_cfg.lr_stage_two,
                                     self.x_cfg.trainable_bg)
        optimizer.init_optimizers()

        self.print(f"-> Painter point Params: {len(renderer.get_point_parameters())}")
        self.print(f"-> Painter color Params: {len(renderer.get_color_parameters())}")
        self.print(f"-> Painter width Params: {len(renderer.get_width_parameters())}")

        self.step = 0  # reset global step
        total_step = self.x_cfg.sds.num_iter
        path_reinit = self.x_cfg.path_reinit

        self.print(f"\ntotal sds optimization steps: {total_step}")
        with tqdm(initial=self.step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_step:
                raster_img = renderer.get_image(step=self.step).to(self.device)

                if self.make_video and (self.step % self.args.framefreq == 0 or self.step == total_step - 1):
                    plot_img(raster_img, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                L_sds, grad = self.diffusion.score_distillation_sampling(
                    raster_img,
                    im_size=self.x_cfg.sds.im_size,
                    prompt=[text_prompt],
                    negative_prompt=[self.args.neg_prompt],
                    guidance_scale=self.x_cfg.sds.guidance_scale,
                    input_augment=self.x_cfg.sds.x_aug,
                    grad_scale=self.x_cfg.sds.grad_scale,
                    t_range=list(self.x_cfg.sds.t_range),
                )

                # Xing Loss for Self-Interaction Problem
                L_add = torch.tensor(0.)
                if self.style == "iconography":
                    L_add = xing_loss_fn(renderer.get_point_parameters()) * self.x_cfg.xing_loss_weight
                # pixel_penalty_loss to combat oversaturation
                if self.style in ["pixelart", "low-poly"]:
                    L_add = pixel_penalty_loss(raster_img) * self.x_cfg.penalty_weight

                loss = L_sds + L_add

                # optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                renderer.clip_curve_shape()

                # re-init paths
                if path_reinit.use and self.step % path_reinit.freq == 0 and self.step < path_reinit.stop_step and self.step != 0:
                    extra_point_params, extra_color_params, extra_width_params = \
                        renderer.reinitialize_paths(f"Step {self.step}",
                                                    self.reinit_dir / f"reinit-{self.step}.svg",
                                                    path_reinit.opacity_threshold,
                                                    path_reinit.area_threshold)
                    optimizer.add_params(extra_point_params, extra_color_params, extra_width_params)

                # update lr
                if self.x_cfg.lr_stage_two.lr_schedule:
                    optimizer.update_lr()

                lr_str = ""
                for k, lr in optimizer.get_lr().items():
                    lr_str += f"{k}_lr: {lr:.4f}, "

                pbar.set_description(
                    lr_str +
                    f"L_total: {loss.item():.4f}, "
                    f"L_add: {L_add.item():.4e}, "
                    f"sds: {grad.item():.5e}"
                )

                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    plot_couple(target_img,
                                raster_img,
                                self.step,
                                prompt=text_prompt,
                                output_dir=self.ft_png_logs_dir.as_posix(),
                                fname=f"iter{self.step}")
                    renderer.pretty_save_svg(self.ft_svg_logs_dir / f"svg_iter{self.step}.svg")

                self.step += 1
                pbar.update(1)

        final_svg_fpth = self.result_path / "finetune_final.svg"
        renderer.pretty_save_svg(final_svg_fpth)

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.result_path / "VF_rendering_stage2.mp4").as_posix()
            ])

        self.close(msg="painterly rendering complete.")

    def load_renderer(self, path_svg=None):
        renderer = Painter(self.args.diffvg,
                           self.style,
                           self.x_cfg.num_segments,
                           self.x_cfg.segment_init,
                           self.x_cfg.radius,
                           self.x_cfg.image_size,
                           self.x_cfg.grid,
                           self.x_cfg.trainable_bg,
                           self.x_cfg.width,
                           path_svg=path_svg,
                           device=self.device)
        return renderer
