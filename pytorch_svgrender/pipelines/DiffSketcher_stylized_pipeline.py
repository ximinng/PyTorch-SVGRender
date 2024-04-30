# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import shutil
import pathlib
from PIL import Image
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets.folder import is_image_file
from tqdm.auto import tqdm
import numpy as np
from skimage.color import rgb2gray
import diffusers

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.libs.metric.lpips_origin import LPIPS
from pytorch_svgrender.libs.metric.piq.perceptual import DISTS as DISTS_PIQ
from pytorch_svgrender.libs.metric.clip_score import CLIPScoreWrapper
from pytorch_svgrender.painter.diffsketcher import (
    Painter, SketchPainterOptimizer, Token2AttnMixinASDSPipeline, Token2AttnMixinASDSSDXLPipeline)
from pytorch_svgrender.painter.diffsketcher.sketch_utils import plt_triplet
from pytorch_svgrender.plt import plot_img
from pytorch_svgrender.painter.diffsketcher.sketch_utils import plt_attn
from pytorch_svgrender.painter.clipasso.sketch_utils import get_mask_u2net, fix_image_scale
from pytorch_svgrender.token2attn.attn_control import AttentionStore, EmptyControl
from pytorch_svgrender.token2attn.ptp_utils import view_images
from pytorch_svgrender.painter.style_clipdraw import sample_indices, StyleLoss, VGG16Extractor
from pytorch_svgrender.diffusers_warp import init_StableDiffusion_pipeline, model2res


class StylizedDiffSketcherPipeline(ModelState):

    def __init__(self, args):
        attn_log_ = ""
        if args.x.attention_init:
            attn_log_ = f"-tk{args.x.token_ind}" \
                        f"{'-XDoG' if args.x.xdog_intersec else ''}" \
                        f"-atc{args.x.attn_coeff}-tau{args.x.softmax_temp}"
        logdir_ = f"sd{args.seed}-im{args.x.image_size}" \
                  f"-ST{args.x.style_strength}" \
                  f"-P{args.x.num_paths}W{args.x.width}{'OP' if args.x.optim_opacity else 'BL'}" \
                  f"{attn_log_}"
        super().__init__(args, log_path_suffix=logdir_)

        # create log dir
        self.png_logs_dir = self.result_path / "png_logs"
        self.svg_logs_dir = self.result_path / "svg_logs"
        self.attn_logs_dir = self.result_path / "attn_logs"
        if self.accelerator.is_main_process:
            self.png_logs_dir.mkdir(parents=True, exist_ok=True)
            self.svg_logs_dir.mkdir(parents=True, exist_ok=True)
            self.attn_logs_dir.mkdir(parents=True, exist_ok=True)

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
            custom_pipeline = Token2AttnMixinASDSSDXLPipeline
            custom_scheduler = diffusers.DPMSolverMultistepScheduler
            self.x_cfg.cross_attn_res = self.x_cfg.cross_attn_res * 2
        elif self.x_cfg.model_id == 'sd21':
            custom_pipeline = Token2AttnMixinASDSPipeline
            custom_scheduler = diffusers.DDIMScheduler
        else:  # sd14, sd15
            custom_pipeline = Token2AttnMixinASDSPipeline
            custom_scheduler = diffusers.DDIMScheduler

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
        )

        self.g_device = torch.Generator(device=self.device).manual_seed(args.seed)

        # init clip model and clip score wrapper
        self.cargs = self.x_cfg.clip
        self.clip_score_fn = CLIPScoreWrapper(self.cargs.model_name,
                                              device=self.device,
                                              visual_score=True,
                                              feats_loss_type=self.cargs.feats_loss_type,
                                              feats_loss_weights=self.cargs.feats_loss_weights,
                                              fc_loss_weight=self.cargs.fc_loss_weight)

        # load STROTSS
        self.style_extractor = VGG16Extractor(space="normal").to(self.device)
        self.style_loss = StyleLoss()

    def extract_ldm_attn(self, prompt):
        # log prompts
        self.print(f"prompt: {prompt}")
        self.print(f"negative_prompt: {self.args.neg_prompt}\n")

        # init controller
        controller = AttentionStore() if self.x_cfg.attention_init else EmptyControl()

        height = width = model2res(self.x_cfg.model_id)
        outputs = self.diffusion(prompt=[prompt],
                                 negative_prompt=[self.args.neg_prompt],
                                 height=height,
                                 width=width,
                                 controller=controller,
                                 num_inference_steps=self.x_cfg.num_inference_steps,
                                 guidance_scale=self.x_cfg.guidance_scale,
                                 generator=self.g_device)

        target_file = self.result_path / "ldm_generated_image.png"
        view_images([np.array(img) for img in outputs.images], save_image=True, fp=target_file)

        if self.x_cfg.attention_init:
            """ldm cross-attention map"""
            cross_attention_maps, tokens = \
                self.diffusion.get_cross_attention([prompt],
                                                   controller,
                                                   res=self.x_cfg.cross_attn_res,
                                                   from_where=("up", "down"),
                                                   save_path=self.result_path / "cross_attn.png")

            self.print(f"the length of tokens is {len(tokens)}, select {self.x_cfg.token_ind}-th token")
            # [res, res, seq_len]
            self.print(f"origin cross_attn_map shape: {cross_attention_maps.shape}")
            # [res, res]
            cross_attn_map = cross_attention_maps[:, :, self.x_cfg.token_ind]
            self.print(f"select cross_attn_map shape: {cross_attn_map.shape}\n")
            cross_attn_map = 255 * cross_attn_map / cross_attn_map.max()
            # [res, res, 3]
            cross_attn_map = cross_attn_map.unsqueeze(-1).expand(*cross_attn_map.shape, 3)
            # [3, res, res]
            cross_attn_map = cross_attn_map.permute(2, 0, 1).unsqueeze(0)
            # [3, clip_size, clip_size]
            cross_attn_map = F.interpolate(cross_attn_map, size=self.x_cfg.image_size, mode='bicubic')
            cross_attn_map = torch.clamp(cross_attn_map, min=0, max=255)
            # rgb to gray
            cross_attn_map = rgb2gray(cross_attn_map.squeeze(0).permute(1, 2, 0)).astype(np.float32)
            # torch to numpy
            if cross_attn_map.shape[-1] != self.x_cfg.image_size and cross_attn_map.shape[-2] != self.x_cfg.image_size:
                cross_attn_map = cross_attn_map.reshape(self.x_cfg.image_size, self.x_cfg.image_size)
            # to [0, 1]
            cross_attn_map = (cross_attn_map - cross_attn_map.min()) / (cross_attn_map.max() - cross_attn_map.min())

            """ldm self-attention map"""
            self_attention_maps, svd, vh_ = \
                self.diffusion.get_self_attention_comp([prompt],
                                                       controller,
                                                       res=self.x_cfg.self_attn_res,
                                                       from_where=("up", "down"),
                                                       img_size=self.x_cfg.image_size,
                                                       max_com=self.x_cfg.max_com,
                                                       save_path=self.result_path)

            # comp self-attention map
            if self.x_cfg.mean_comp:
                self_attn = np.mean(vh_, axis=0)
                self.print(f"use the mean of {self.x_cfg.max_com} comps.")
            else:
                self_attn = vh_[self.x_cfg.comp_idx]
                self.print(f"select {self.x_cfg.comp_idx}-th comp.")
            # to [0, 1]
            self_attn = (self_attn - self_attn.min()) / (self_attn.max() - self_attn.min())
            # visual final self-attention
            self_attn_vis = np.copy(self_attn)
            self_attn_vis = self_attn_vis * 255
            self_attn_vis = np.repeat(np.expand_dims(self_attn_vis, axis=2), 3, axis=2).astype(np.uint8)
            self_attn_vis = Image.fromarray(self_attn_vis)
            self_attn_vis = np.array(self_attn_vis)
            view_images(self_attn_vis, save_image=True, fp=self.result_path / "self-attn-final.png")

            """attention map fusion"""
            attn_map = self.x_cfg.attn_coeff * cross_attn_map + (1 - self.x_cfg.attn_coeff) * self_attn
            # to [0, 1]
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

            self.print(f"-> fusion attn_map: {attn_map.shape}")
        else:
            attn_map = None

        return target_file.as_posix(), attn_map

    def load_render(self, target_img, attention_map, mask=None):
        renderer = Painter(self.x_cfg,
                           self.args.diffvg,
                           num_strokes=self.x_cfg.num_paths,
                           num_segments=self.x_cfg.num_segments,
                           canvas_size=self.x_cfg.image_size,
                           device=self.device,
                           target_im=target_img,
                           attention_map=attention_map,
                           mask=mask)
        return renderer

    @property
    def clip_norm_(self):
        return transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def clip_pair_augment(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          im_res: int,
                          augments: str = "affine_norm",
                          num_aug: int = 4):
        # init augmentations
        augment_list = []
        if "affine" in augments:
            augment_list.append(
                transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5)
            )
            augment_list.append(
                transforms.RandomResizedCrop(im_res, scale=(0.8, 0.8), ratio=(1.0, 1.0))
            )
        augment_list.append(self.clip_norm_)  # CLIP Normalize

        # compose augmentations
        augment_compose = transforms.Compose(augment_list)
        # make augmentation pairs
        x_augs, y_augs = [self.clip_score_fn.normalize(x)], [self.clip_score_fn.normalize(y)]
        # repeat N times
        for n in range(num_aug):
            augmented_pair = augment_compose(torch.cat([x, y]))
            x_augs.append(augmented_pair[0].unsqueeze(0))
            y_augs.append(augmented_pair[1].unsqueeze(0))
        xs = torch.cat(x_augs, dim=0)
        ys = torch.cat(y_augs, dim=0)
        return xs, ys

    def painterly_rendering(self, prompt, style_fpath):
        # init attention
        target_file, attention_map = self.extract_ldm_attn(prompt)

        timesteps_ = self.diffusion.scheduler.timesteps.cpu().numpy().tolist()
        self.print(f"{len(timesteps_)} denoising steps, {timesteps_}")

        perceptual_loss_fn = None
        if self.x_cfg.perceptual.coeff > 0:
            if self.x_cfg.perceptual.name == "lpips":
                lpips_loss_fn = LPIPS(net=self.x_cfg.perceptual.lpips_net).to(self.device)
                perceptual_loss_fn = partial(lpips_loss_fn.forward, return_per_layer=False, normalize=False)
            elif self.x_cfg.perceptual.name == "dists":
                perceptual_loss_fn = DISTS_PIQ()

        style_img, feat_style = self.load_and_process_style_file(style_fpath)

        inputs, mask = self.get_target(target_file,
                                       self.x_cfg.image_size,
                                       self.result_path,
                                       self.x_cfg.u2net_path,
                                       self.x_cfg.mask_object,
                                       self.x_cfg.fix_scale,
                                       self.device)
        inputs = inputs.detach()  # inputs as GT
        self.print("inputs shape: ", inputs.shape)

        # load renderer
        renderer = self.load_render(inputs, attention_map, mask=mask)
        # init img
        img = renderer.init_image(stage=0)
        self.print("init_image shape: ", img.shape)
        plot_img(img, self.result_path, fname="init_sketch")
        # load optimizer
        optimizer = SketchPainterOptimizer(renderer,
                                           self.x_cfg.lr,
                                           self.x_cfg.optim_opacity,
                                           self.x_cfg.optim_rgba,
                                           self.x_cfg.color_lr,
                                           self.x_cfg.optim_width,
                                           self.x_cfg.width_lr)
        optimizer.init_optimizers()

        # log params
        self.print(f"-> Painter point Params: {len(renderer.get_points_params())}")
        self.print(f"-> Painter width Params: {len(renderer.get_width_parameters())}")
        self.print(f"-> Painter color Params: {len(renderer.get_color_parameters())}")

        total_iter = self.x_cfg.num_iter
        best_visual_loss, best_semantic_loss = 100, 100
        min_delta = 1e-6

        self.print(f"\ntotal optimization steps: {total_iter}")
        with tqdm(initial=self.step, total=total_iter, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_iter:
                raster_sketch = renderer.get_image().to(self.device)

                if self.make_video and (self.step % self.args.framefreq == 0 or self.step == total_iter - 1):
                    plot_img(raster_sketch, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                # ASDS loss
                sds_loss, grad = torch.tensor(0), torch.tensor(0)
                if self.step >= self.x_cfg.sds.warmup:
                    grad_scale = self.x_cfg.sds.grad_scale if self.step > self.x_cfg.sds.warmup else 0
                    sds_loss, grad = self.diffusion.score_distillation_sampling(
                        raster_sketch,
                        crop_size=self.x_cfg.sds.crop_size,
                        augments=self.x_cfg.sds.augmentations,
                        prompt=[prompt],
                        negative_prompt=self.args.neg_prompt,
                        guidance_scale=self.x_cfg.sds.guidance_scale,
                        grad_scale=grad_scale,
                        t_range=list(self.x_cfg.sds.t_range),
                    )

                # CLIP data augmentation
                raster_sketch_aug, inputs_aug = self.clip_pair_augment(
                    raster_sketch, inputs,
                    im_res=224,
                    augments=self.cargs.augmentations,
                    num_aug=self.cargs.num_aug
                )

                # clip visual loss
                total_visual_loss = torch.tensor(0)
                l_clip_fc, l_clip_conv, clip_conv_loss_sum = torch.tensor(0), [], torch.tensor(0)
                if self.x_cfg.clip.vis_loss > 0:
                    l_clip_fc, l_clip_conv = self.clip_score_fn.compute_visual_distance(
                        raster_sketch_aug, inputs_aug, clip_norm=False
                    )
                    clip_conv_loss_sum = sum(l_clip_conv)
                    total_visual_loss = self.x_cfg.clip.vis_loss * (clip_conv_loss_sum + l_clip_fc)

                # text-visual loss
                l_tvd = torch.tensor(0.)
                if self.cargs.text_visual_coeff > 0:
                    l_tvd = self.clip_score_fn.compute_text_visual_distance(
                        raster_sketch_aug, prompt
                    ) * self.cargs.text_visual_coeff

                # perceptual loss
                l_percep = torch.tensor(0.)
                if perceptual_loss_fn is not None:
                    l_perceptual = perceptual_loss_fn(raster_sketch, inputs).mean()
                    l_percep = l_perceptual * self.x_cfg.perceptual.coeff

                # style loss
                feat_content = self.style_extractor(raster_sketch)
                xx, xy = sample_indices(feat_content[0], feat_style)
                np.random.shuffle(xx)
                np.random.shuffle(xy)
                l_style = self.x_cfg.style_strength * self.style_loss.forward(
                    feat_content, feat_content, feat_style, [xx, xy], 0
                )

                # total loss
                loss = sds_loss + total_visual_loss + l_tvd + l_percep + l_style

                # optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                # update lr
                if self.x_cfg.lr_schedule:
                    optimizer.update_lr(self.step, self.x_cfg.decay_steps)

                # records
                pbar.set_description(
                    f"lr: {optimizer.get_lr():.2f}, "
                    f"l_total: {loss.item():.4f}, "
                    f"l_clip_fc: {l_clip_fc.item():.4f}, "
                    f"l_clip_conv({len(l_clip_conv)}): {clip_conv_loss_sum.item():.4f}, "
                    f"l_tvd: {l_tvd.item():.4f}, "
                    f"l_percep: {l_percep.item():.4f}, "
                    f"l_style: {l_style.item():.4f}, "
                    f"sds: {grad.item():.4e}"
                )

                # log raster and svg
                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    # log png
                    plt_triplet(inputs,
                                raster_sketch,
                                style_img,
                                self.step,
                                prompt,
                                output_dir=self.png_logs_dir.as_posix(),
                                fname=f"iter{self.step}")
                    # log svg
                    renderer.save_svg(self.svg_logs_dir.as_posix(), f"svg_iter{self.step}")
                    # log cross attn
                    if self.x_cfg.log_cross_attn:
                        controller = AttentionStore()
                        _, _ = self.diffusion.get_cross_attention([prompt],
                                                                  controller,
                                                                  res=self.x_cfg.cross_attn_res,
                                                                  from_where=("up", "down"),
                                                                  save_path=self.attn_logs_dir / f"iter{self.step}.png")

                # logging the best raster images and SVG
                if self.step % self.args.eval_step == 0 and self.accelerator.is_main_process:
                    with torch.no_grad():
                        # visual metric
                        l_clip_fc, l_clip_conv = self.clip_score_fn.compute_visual_distance(
                            raster_sketch_aug, inputs_aug, clip_norm=False
                        )
                        loss_eval = sum(l_clip_conv) + l_clip_fc

                        cur_delta = loss_eval.item() - best_visual_loss
                        if abs(cur_delta) > min_delta and cur_delta < 0:
                            best_visual_loss = loss_eval.item()
                            best_iter_v = self.step
                            plt_triplet(inputs,
                                        raster_sketch,
                                        style_img,
                                        best_iter_v,
                                        prompt,
                                        output_dir=self.result_path.as_posix(),
                                        fname="visual_best")
                            renderer.save_svg(self.result_path.as_posix(), "visual_best")

                        # semantic metric
                        loss_eval = self.clip_score_fn.compute_text_visual_distance(
                            raster_sketch_aug, prompt
                        )
                        cur_delta = loss_eval.item() - best_semantic_loss
                        if abs(cur_delta) > min_delta and cur_delta < 0:
                            best_semantic_loss = loss_eval.item()
                            best_iter_s = self.step
                            plt_triplet(inputs,
                                        raster_sketch,
                                        style_img,
                                        best_iter_s,
                                        prompt,
                                        output_dir=self.result_path.as_posix(),
                                        fname="semantic_best")
                            renderer.save_svg(self.result_path.as_posix(), "semantic_best")

                # log attention, for once
                if self.step == 0 and self.x_cfg.attention_init and self.accelerator.is_main_process:
                    plt_attn(renderer.get_attn(),
                             renderer.get_thresh(),
                             inputs,
                             renderer.get_inds(),
                             (self.result_path / "attention_map.png").as_posix())

                self.step += 1
                pbar.update(1)

        # saving final result
        renderer.save_svg(self.result_path.as_posix(), f"final_best_step")

        final_raster_sketch = renderer.get_image().to(self.device)
        plot_img(final_raster_sketch,
                 output_dir=self.result_path,
                 fname='final_best_step')

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.result_path / "stylediffsketcher_rendering.mp4").as_posix()
            ])

        self.close(msg="painterly rendering complete.")

    def load_and_process_style_file(self, style_fpath):
        # load style file
        style_path = Path(style_fpath)
        assert style_path.exists(), f"{style_fpath} is not exist!"
        style_img = self.style_file_preprocess(style_path.as_posix())
        self.print(f"load style file from: {style_path.as_posix()}")
        shutil.copy(style_fpath, self.result_path)  # copy style file

        # extract style features from style image
        feat_style = None
        for i in range(5):
            with torch.no_grad():
                # r is region of interest (mask)
                feat_e = self.style_extractor.forward_samples_hypercolumn(style_img, samps=1000)
                feat_style = feat_e if feat_style is None else torch.cat((feat_style, feat_e), dim=2)

        return style_img, feat_style

    def style_file_preprocess(self, style_path):
        process_comp = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: t - 0.5),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
            # transforms.Lambda(lambda t: (t + 1) / 2),
        ])

        style_pil = Image.open(style_path).convert("RGB")  # open file
        style_file = process_comp(style_pil)  # preprocess
        style_file = style_file.to(self.device)
        return style_file

    def get_target(self,
                   target_file,
                   image_size,
                   output_dir,
                   u2net_path,
                   mask_object,
                   fix_scale,
                   device):
        if not is_image_file(target_file):
            raise TypeError(f"{target_file} is not image file.")

        target = Image.open(target_file)

        if target.mode == "RGBA":
            # Create a white rgba background
            new_image = Image.new("RGBA", target.size, "WHITE")
            # Paste the image on the background.
            new_image.paste(target, (0, 0), target)
            target = new_image
        target = target.convert("RGB")

        # U2Net mask
        mask = target
        if mask_object:
            if pathlib.Path(u2net_path).exists():
                masked_im, mask = get_mask_u2net(target, output_dir, u2net_path, device)
                target = masked_im
            else:
                self.print(f"'{u2net_path}' is not exist, disable mask target")

        if fix_scale:
            target = fix_image_scale(target)

        if fix_scale:
            target = fix_image_scale(target)

        # define image transforms
        transforms_ = []
        if target.size[0] != target.size[1]:
            transforms_.append(transforms.Resize((image_size, image_size)))
        else:
            transforms_.append(transforms.Resize(image_size))
            transforms_.append(transforms.CenterCrop(image_size))
        transforms_.append(transforms.ToTensor())

        # preprocess
        data_transforms = transforms.Compose(transforms_)
        target_ = data_transforms(target).unsqueeze(0).to(self.device)

        return target_, mask
