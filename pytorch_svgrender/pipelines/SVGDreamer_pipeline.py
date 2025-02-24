# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import pathlib
from PIL import Image
from typing import AnyStr

import numpy as np
from tqdm.auto import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR
import torchvision
from torchvision import transforms

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.libs.solver.optim import get_optimizer
from pytorch_svgrender.painter.svgdreamer import Painter, PainterOptimizer
from pytorch_svgrender.painter.svgdreamer.painter_params import CosineWithWarmupLRLambda
from pytorch_svgrender.painter.live import xing_loss_fn
from pytorch_svgrender.painter.svgdreamer import VectorizedParticleSDSPipeline
from pytorch_svgrender.plt import plot_img
from pytorch_svgrender.utils.color_attrs import init_tensor_with_color
from pytorch_svgrender.token2attn.ptp_utils import view_images
from pytorch_svgrender.model_helper import model2res

import ImageReward as RM


class SVGDreamerPipeline(ModelState):

    def __init__(self, args):
        assert args.x.style in ["iconography", "pixelart", "low-poly", "painting", "sketch", "ink"]
        assert args.x.guidance.n_particle >= args.x.guidance.vsd_n_particle
        assert args.x.guidance.n_particle >= args.x.guidance.phi_n_particle
        assert args.x.guidance.n_phi_sample >= 1

        logdir_ = f"sd{args.seed}" \
                  f"-{'vpsd' if args.x.skip_sive else 'sive'}" \
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
        self.init_stage_two_dir = self.result_path / "stage_two_init_logs"
        self.phi_samples_dir = self.result_path / "phi_sampling_logs"

        if self.accelerator.is_main_process:
            self.png_logs_dir.mkdir(parents=True, exist_ok=True)
            self.svg_logs_dir.mkdir(parents=True, exist_ok=True)
            self.ft_png_logs_dir.mkdir(parents=True, exist_ok=True)
            self.ft_svg_logs_dir.mkdir(parents=True, exist_ok=True)
            self.sd_sample_dir.mkdir(parents=True, exist_ok=True)
            self.reinit_dir.mkdir(parents=True, exist_ok=True)
            self.init_stage_two_dir.mkdir(parents=True, exist_ok=True)
            self.phi_samples_dir.mkdir(parents=True, exist_ok=True)

        self.select_fpth = self.result_path / 'select_sample.png'

        # make video log
        self.make_video = self.args.mv
        if self.make_video:
            self.frame_idx = 0
            self.frame_log_dir = self.result_path / "frame_logs"
            self.frame_log_dir.mkdir(parents=True, exist_ok=True)

        self.g_device = torch.Generator(device=self.device).manual_seed(args.seed)

        self.pipeline = VectorizedParticleSDSPipeline(args, args.diffuser, self.x_cfg.guidance, self.device)

        # load reward model
        self.reward_model = None
        if self.x_cfg.guidance.phi_ReFL:
            self.reward_model = RM.load("ImageReward-v1.0", device=self.device, download_root=self.x_cfg.reward_path)

        self.style = self.x_cfg.style
        if self.style == "pixelart":
            self.x_cfg.lr_stage_one.lr_schedule = False
            self.x_cfg.lr_stage_two.lr_schedule = False

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

    def SIVE_stage(self, text_prompt: str):
        # TODO: SIVE implementation
        pass

    def painterly_rendering(self, text_prompt: str, target_file: AnyStr = None):
        # log prompts
        self.print(f"prompt: {text_prompt}")
        self.print(f"neg_prompt: {self.args.neg_prompt}\n")

        # for convenience
        im_size = self.x_cfg.image_size
        guidance_cfg = self.x_cfg.guidance
        n_particle = self.x_cfg.guidance.n_particle
        total_step = self.x_cfg.guidance.num_iter
        path_reinit = self.x_cfg.path_reinit

        init_from_target = True if (target_file and pathlib.Path(target_file).exists()) else False
        # switch mode
        if self.x_cfg.skip_sive and not init_from_target:
            # mode 1: optimization with VPSD from scratch
            # randomly init
            self.print("optimization with VPSD from scratch...")
            if self.x_cfg.color_init == 'rand':
                target_img = torch.randn(1, 3, im_size, im_size)
                self.print("color: randomly init")
            else:
                target_img = init_tensor_with_color(self.x_cfg.color_init, 1, im_size, im_size)
                self.print(f"color: {self.x_cfg.color_init}")

            # log init target_img
            plot_img(target_img, self.result_path, fname='init_target_img')
            final_svg_path = None
        elif init_from_target:
            # mode 2: load the SVG file and finetune it
            self.print(f"load svg from {target_file} ...")
            self.print(f"SVG fine-tuning via VPSD...")
            final_svg_path = target_file
            if self.x_cfg.color_init == 'target_randn':
                # special order: init newly paths color use random color
                target_img = torch.randn(1, 3, im_size, im_size)
                self.print("color: randomly init")
            else:
                # load the SVG and init newly paths color use target_img
                # note: the target will be converted to png via pydiffvg when load_renderer called
                target_img = None
        else:
            # mode 3: text-to-img-to-svg (two stage)
            target_img, final_svg_path = self.SIVE_stage(text_prompt)
            self.x_cfg.path_svg = final_svg_path
            self.print("\n SVG fine-tuning via VPSD...")
            plot_img(target_img, self.result_path, fname='init_target_img')

        # create svg renderer
        renderers = [self.load_renderer(final_svg_path) for _ in range(n_particle)]

        # randomly initialize the particles
        if self.x_cfg.skip_sive or init_from_target:
            if target_img is None:
                target_img = self.target_file_preprocess(self.result_path / 'target_img.png')
            for render in renderers:
                render.component_wise_path_init(gt=target_img, pred=None, init_type='random')

        # log init images
        for i, r in enumerate(renderers):
            init_imgs = r.init_image(stage=0, num_paths=self.x_cfg.num_paths)
            plot_img(init_imgs, self.init_stage_two_dir, fname=f"init_img_stage_two_{i}")

        # init renderer optimizer
        optimizers = []
        for renderer in renderers:
            optim_ = PainterOptimizer(renderer,
                                      self.style,
                                      guidance_cfg.num_iter,
                                      self.x_cfg.lr_stage_two,
                                      self.x_cfg.trainable_bg)
            optim_.init_optimizers()
            optimizers.append(optim_)

        # init phi_model optimizer
        phi_optimizer = get_optimizer('adamW',
                                      self.pipeline.phi_params,
                                      guidance_cfg.phi_lr,
                                      guidance_cfg.phi_optim)
        # init phi_model lr scheduler
        phi_scheduler = None
        schedule_cfg = guidance_cfg.phi_schedule
        if schedule_cfg.use:
            phi_lr_lambda = CosineWithWarmupLRLambda(num_steps=schedule_cfg.total_step,
                                                     warmup_steps=schedule_cfg.warmup_steps,
                                                     warmup_start_lr=schedule_cfg.warmup_start_lr,
                                                     warmup_end_lr=schedule_cfg.warmup_end_lr,
                                                     cosine_end_lr=schedule_cfg.cosine_end_lr)
            phi_scheduler = LambdaLR(phi_optimizer, lr_lambda=phi_lr_lambda, last_epoch=-1)

        self.print(f"-> Painter point Params: {len(renderers[0].get_point_parameters())}")
        self.print(f"-> Painter color Params: {len(renderers[0].get_color_parameters())}")
        self.print(f"-> Painter width Params: {len(renderers[0].get_width_parameters())}")

        L_reward = torch.tensor(0.)

        self.step = 0  # reset global step
        self.print(f"\ntotal VPSD optimization steps: {total_step}")
        with tqdm(initial=self.step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_step:
                # set particles
                particles = [renderer.get_image() for renderer in renderers]
                raster_imgs = torch.cat(particles, dim=0)

                if self.make_video and (self.step % self.args.framefreq == 0 or self.step == total_step - 1):
                    plot_img(raster_imgs, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                L_guide, grad, latents, t_step = self.pipeline.variational_score_distillation(
                    raster_imgs.to(self.weight_dtype),
                    self.step,
                    prompt=[text_prompt],
                    negative_prompt=self.args.neg_prompt,
                    grad_scale=guidance_cfg.grad_scale,
                    enhance_particle=guidance_cfg.particle_aug,
                    im_size=model2res(self.x_cfg.model_id)
                )

                # Xing Loss for Self-Interaction Problem
                L_add = torch.tensor(0.)
                if self.style == "iconography" or self.x_cfg.xing_loss.use:
                    for r in renderers:
                        L_add += xing_loss_fn(r.get_point_parameters()) * self.x_cfg.xing_loss.weight

                loss = L_guide + L_add

                # optimization
                for opt_ in optimizers:
                    opt_.zero_grad_()
                loss.backward()
                for opt_ in optimizers:
                    opt_.step_()

                # phi_model optimization
                for _ in range(guidance_cfg.phi_update_step):
                    L_lora = self.pipeline.train_phi_model(latents, guidance_cfg.phi_t, as_latent=True)

                    phi_optimizer.zero_grad()
                    L_lora.backward()
                    phi_optimizer.step()

                # reward learning
                if guidance_cfg.phi_ReFL and self.step % guidance_cfg.phi_sample_step == 0:
                    with torch.no_grad():
                        phi_outputs = []
                        phi_sample_paths = []
                        for idx in range(guidance_cfg.n_phi_sample):
                            phi_output = self.pipeline.sample(text_prompt,
                                                              num_inference_steps=guidance_cfg.phi_infer_step,
                                                              generator=self.g_device)
                            sample_path = (self.phi_samples_dir / f'iter{idx}.png').as_posix()
                            phi_output.images[0].save(sample_path)
                            phi_sample_paths.append(sample_path)

                            phi_output_np = np.array(phi_output.images[0])
                            phi_outputs.append(phi_output_np)
                        # save all samples
                        view_images(phi_outputs, save_image=True,
                                    num_rows=max(len(phi_outputs) // 6, 1),
                                    fp=self.phi_samples_dir / f'samples_iter{self.step}.png')

                    ranking, rewards = self.reward_model.inference_rank(text_prompt, phi_sample_paths)
                    self.print(f"ranking: {ranking}, reward score: {rewards}")

                    for k in range(guidance_cfg.n_phi_sample):
                        phi = self.target_file_preprocess(phi_sample_paths[ranking[k] - 1])
                        L_reward = self.pipeline.train_phi_model_refl(phi, weight=rewards[k])

                        phi_optimizer.zero_grad()
                        L_reward.backward()
                        phi_optimizer.step()

                # update the learning rate of the phi_model
                if phi_scheduler is not None:
                    phi_scheduler.step()

                # curve regularization
                for r in renderers:
                    r.clip_curve_shape()

                # re-init paths
                if path_reinit.use and self.step % path_reinit.freq == 0 and self.step < path_reinit.stop_step and self.step != 0:
                    for i, r in enumerate(renderers):
                        extra_point_params, extra_color_params, extra_width_params = \
                            r.reinitialize_paths(f"P{i} - Step {self.step}",
                                                 self.reinit_dir / f"reinit-{self.step}_p{i}.svg",
                                                 path_reinit.opacity_threshold,
                                                 path_reinit.area_threshold)
                        optimizers[i].add_params(extra_point_params, extra_color_params, extra_width_params)

                # update lr
                if self.x_cfg.lr_stage_two.lr_schedule:
                    for opt_ in optimizers:
                        opt_.update_lr()

                # log pretrained model lr
                lr_str = ""
                for k, lr in optimizers[0].get_lr().items():
                    lr_str += f"{k}_lr: {lr:.4f}, "
                # log phi model lr
                cur_phi_lr = phi_optimizer.param_groups[0]['lr']
                lr_str += f"phi_lr: {cur_phi_lr:.3e}, "

                pbar.set_description(
                    lr_str +
                    f"t: {t_step.item():.2f}, "
                    f"L_total: {loss.item():.4f}, "
                    f"L_add: {L_add.item():.4e}, "
                    f"L_lora: {L_lora.item():.4f}, "
                    f"L_reward: {L_reward.item():.4f}, "
                    f"vpsd: {grad.item():.4e}"
                )

                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    # save png
                    torchvision.utils.save_image(raster_imgs,
                                                 fp=self.ft_png_logs_dir / f'iter{self.step}.png')

                    # save svg
                    for i, r in enumerate(renderers):
                        r.pretty_save_svg(self.ft_svg_logs_dir / f"svg_iter{self.step}_p{i}.svg")

                self.step += 1
                pbar.update(1)

        # save final
        for i, r in enumerate(renderers):
            final_svg_path = self.result_path / f"finetune_final_p_{i}.svg"
            r.pretty_save_svg(final_svg_path)
        # save SVGs
        torchvision.utils.save_image(raster_imgs, fp=self.result_path / f'all_particles.png')

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.result_path / "svgdreamer_rendering.mp4").as_posix()
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

        # if load a svg file, then rasterize it
        save_path = self.result_path / 'target_img.png'
        if path_svg is not None and (not save_path.exists()):
            canvas_width, canvas_height, shapes, shape_groups = renderer.load_svg(path_svg)
            render_img = renderer.render_image(canvas_width, canvas_height, shapes, shape_groups)
            torchvision.utils.save_image(render_img, fp=save_path)
        return renderer
