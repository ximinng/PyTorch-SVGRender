# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import re
from typing import Any, List, Optional, Union, Dict
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg, StableDiffusionPipelineOutput)
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from pytorch_svgrender.diffusers_warp import init_StableDiffusion_pipeline, init_diffusers_unet


class VectorizedParticleSDSPipeline(torch.nn.Module):

    def __init__(self, args: DictConfig, diffuser_cfg: DictConfig, guidance_cfg: DictConfig, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        assert guidance_cfg.n_particle >= guidance_cfg.vsd_n_particle
        assert guidance_cfg.n_particle >= guidance_cfg.phi_n_particle

        pipe_kwargs = {
            "device": self.device,
            "torch_dtype": torch.float16 if args.state.mprec == 'fp16' else torch.float32,
            "local_files_only": not diffuser_cfg.download,
            "force_download": diffuser_cfg.force_download,
            "resume_download": diffuser_cfg.resume_download,
            "ldm_speed_up": args.x.ldm_speed_up,
            "enable_xformers": args.x.enable_xformers,
            "gradient_checkpoint": args.x.gradient_checkpoint,
            "cpu_offload": args.x.cpu_offload,
            "vae_slicing": False
        }

        # load pretrained model
        self.sd_pipeline = init_StableDiffusion_pipeline(
            args.x.model_id,
            custom_pipeline=StableDiffusionPipeline,
            custom_scheduler=DDIMScheduler,
            **pipe_kwargs
        )
        # disable grads
        self.sd_pipeline.vae.requires_grad_(False)
        self.sd_pipeline.text_encoder.requires_grad_(False)
        self.sd_pipeline.unet.requires_grad_(False)
        # set components
        self.vae = self.sd_pipeline.vae
        self.unet = self.sd_pipeline.unet
        self.scheduler = self.sd_pipeline.scheduler
        self.tokenizer = self.sd_pipeline.tokenizer
        self.text_encoder = self.sd_pipeline.text_encoder

        if guidance_cfg.phi_model == 'lora':
            if guidance_cfg.phi_single:  # default, use the single unet
                # load LoRA model from the pretrained model
                unet_ = self.unet
            else:
                # create a new unet model
                pipe_kwargs.pop('cpu_offload')
                pipe_kwargs.pop('vae_slicing')
                unet_ = init_diffusers_unet(args.x.model_id, **pipe_kwargs)

            # set correct LoRA layers
            self.unet_phi, phi_model_layers = self.set_lora_layers(unet_)
            self.phi_params = list(phi_model_layers.parameters())
            self.lora_cross_attention_kwargs = {"scale": guidance_cfg.lora_attn_scale} \
                if guidance_cfg.use_attn_scale else {}
            self.vae_phi = self.vae
            self.vae_phi.requires_grad_(False)

        elif guidance_cfg.phi_model == 'unet_simple':
            self.unet_phi = UNet2DConditionModel(
                sample_size=64,
                in_channels=4,
                out_channels=4,
                layers_per_block=1,
                block_out_channels=(128, 256, 384, 512),
                down_block_types=(
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                ),
                up_block_types=(
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                ),
                cross_attention_dim=self.unet.config.cross_attention_dim
            ).to(device)
            self.phi_params = list(self.unet_phi.parameters())
            self.vae_phi = self.vae
            # reset lora
            guidance_cfg.use_attn_scale = False
            guidance_cfg.lora_attn_scale = False

        # hyper-params
        self.phi_single = guidance_cfg.phi_single
        self.guidance_scale: float = guidance_cfg.guidance_scale
        self.guidance_scale_lora: float = guidance_cfg.phi_guidance_scale
        self.grad_clip_val: Union[float, None] = guidance_cfg.grad_clip_val
        self.vsd_n_particle: int = guidance_cfg.vsd_n_particle
        self.phi_n_particle: int = guidance_cfg.phi_n_particle
        self.t_schedule: str = guidance_cfg.t_schedule
        self.t_range = list(guidance_cfg.t_range)
        print(
            f"diffusion torch_dtype: {pipe_kwargs['torch_dtype']} \n",
            f'n_particles: {guidance_cfg.n_particle}, '
            f'enhance_particles: {guidance_cfg.particle_aug}, '
            f'n_particles of score: {self.vsd_n_particle}, '
            f'n_particles of phi_model: {self.phi_n_particle}, \n'
            f't_range: {self.t_range}, '
            f't_schedule: {self.t_schedule}, \n'
            f'guidance_scale: {self.guidance_scale}, phi_guidance_scale: {self.guidance_scale_lora}.'
        )
        print(f"phi_model: {guidance_cfg.phi_model}, "
              f"use lora_cross_attn: {guidance_cfg.use_attn_scale}, "
              f"lora_attn_scale: {guidance_cfg.lora_attn_scale}. \n")

        # for convenience
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.text_embeddings = None
        self.text_embedd_cond, self.text_embedd_uncond = None, None
        self.text_embeddings_phi = None
        self.t = None

    def set_lora_layers(self, unet):  # set correct lora layers
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") \
                else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim
            ).to(self.device)
        unet.set_attn_processor(lora_attn_procs)
        lora_layers = AttnProcsLayers(unet.attn_processors)

        unet.requires_grad_(False)
        for param in lora_layers.parameters():
            param.requires_grad_(True)
        return unet, lora_layers

    @torch.no_grad()
    def encode_prompt(self,
                      prompt,
                      device,
                      do_classifier_free_guidance,
                      negative_prompt=None):
        # text conditional embed
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(device))[0]

        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            # unconditional embed
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device))[0]

            concat_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            return concat_prompt_embeds, negative_prompt_embeds, prompt_embeds

        return prompt_embeds, None, None

    def sampling(self,
                 vae,
                 unet,
                 scheduler,
                 prompt: Union[str, List[str]] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 negative_prompt: Optional[Union[str, List[str]]] = None,
                 num_images_per_prompt: Optional[int] = 1,
                 eta: float = 0.0,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 latents: Optional[torch.FloatTensor] = None,
                 output_type: Optional[str] = "pil",
                 return_dict: bool = True,
                 cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                 guidance_rescale: float = 0.0):
        # 0. Default height and width to unet
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = height or unet.config.sample_size * vae_scale_factor
        width = width or unet.config.sample_size * vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, _, _ = self.encode_prompt(
            prompt,
            self.device,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # 4. Prepare timesteps
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = unet.config.in_channels
        latents = self.sd_pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.sd_pipeline.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.sd_pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # update progress_bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.sd_pipeline.run_safety_checker(image, self.device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.sd_pipeline.image_processor.postprocess(image, output_type=output_type,
                                                             do_denormalize=do_denormalize)
        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def sample(self,
               prompt,
               height: Optional[int] = None,
               width: Optional[int] = None,
               num_inference_steps: int = 50,
               generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
               output_type: Optional[str] = "pil"):
        return self.sampling(self.vae, self.unet, self.scheduler,
                             prompt=prompt,
                             height=height, width=width,
                             num_inference_steps=num_inference_steps,
                             guidance_scale=self.guidance_scale,
                             generator=generator,
                             output_type=output_type)

    def sample_lora(self,
                    prompt,
                    height: Optional[int] = None,
                    width: Optional[int] = None,
                    num_inference_steps: int = 50,
                    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                    output_type: Optional[str] = "pil"):
        return self.sampling(self.vae_phi, self.unet_phi, self.scheduler,
                             prompt=prompt,
                             height=height, width=width,
                             num_inference_steps=num_inference_steps,
                             guidance_scale=self.guidance_scale_lora,
                             generator=generator,
                             cross_attention_kwargs=self.lora_cross_attention_kwargs,
                             output_type=output_type)

    def encode2latent(self, images):
        images = (2 * images - 1).clamp(-1.0, 1.0)  # images: [B, 3, H, W]
        # encode images
        latents = self.vae.encode(images).latent_dist.sample()
        latents = self.vae.config.scaling_factor * latents
        return latents

    def get_noise_map(self, noise_pred, guidance_scale=7.5, use_cfg=True):
        if use_cfg:
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_map = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
            return noise_map
        else:
            return noise_pred

    def train_phi_model(self,
                        pred_rgb: torch.Tensor,
                        new_timesteps: bool = False,
                        as_latent: bool = False):
        # interp to 512x512 to be fed into vae.
        if as_latent:
            latents = pred_rgb
        else:
            pred_rgb_ = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode2latent(pred_rgb_)

        # get phi particles
        indices = torch.randperm(latents.size(0))
        latents_phi = latents[indices[:self.phi_n_particle]]
        latents_phi = latents_phi.detach()

        # get timestep
        if new_timesteps:
            t = torch.randint(0, self.num_train_timesteps, (1,), device=self.device)
        else:
            t = self.t

        noise = torch.randn_like(latents_phi)
        noisy_latents = self.scheduler.add_noise(latents_phi, noise, t)

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents_phi, noise, t)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        # predict the noise residual and compute loss
        noise_pred = self.unet_phi(
            noisy_latents, t,
            encoder_hidden_states=self.text_embeddings_phi,
            cross_attention_kwargs=self.lora_cross_attention_kwargs,
        ).sample

        return F.mse_loss(noise_pred, target, reduction="mean")

    def train_phi_model_refl(self,
                             pred_rgb: torch.Tensor,
                             weight: float = 1,
                             new_timesteps: bool = True):
        # interp to 512x512 to be fed into vae.
        pred_rgb_ = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # encode image into latents with vae, requires grad!
        latents = self.encode2latent(pred_rgb_)

        # get phi particles
        indices = torch.randperm(latents.size(0))
        latents_phi = latents[indices[:self.phi_n_particle]]
        latents_phi = latents_phi.detach()

        # get timestep
        if new_timesteps:
            t = torch.randint(0, self.num_train_timesteps, (1,), device=self.device)
        else:
            t = self.t

        noise = torch.randn_like(latents_phi)
        noisy_latents = self.scheduler.add_noise(latents_phi, noise, t)

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents_phi, noise, t)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        # predict the noise residual and compute loss
        noise_pred = self.unet_phi(
            noisy_latents, t,
            encoder_hidden_states=self.text_embedd_cond,
            cross_attention_kwargs=self.lora_cross_attention_kwargs,
        ).sample

        rewards = torch.tensor(weight, dtype=torch.float32, device=self.device)
        return rewards * F.mse_loss(noise_pred, target, reduction="mean")

    def schedule_timestep(self, step):
        min_step = int(self.num_train_timesteps * self.t_range[0])
        max_step = int(self.num_train_timesteps * self.t_range[1])
        if self.t_schedule == 'randint':
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        elif re.match(r"max_([\d.]+)_(\d+)", self.t_schedule):
            # Anneal time schedule
            # e.g: t_schedule == 'max_0.5_200'
            # [0.02, 0.98] -> [0.02, 0.5] after 200 steps
            tag, t_val, step_upd = str(self.t_schedule).split('_')
            t_val, step_upd = float(t_val), int(step_upd)
            if step >= step_upd:
                max_step = int(self.num_train_timesteps * t_val)
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        elif re.match(r"min_([\d.]+)_(\d+)", self.t_schedule):
            # Anneal time schedule
            # e.g: t_schedule == 'min_0.5_200'
            # [0.02, 0.98] -> [0.5, 0.98] after 200 steps
            tag, t_val, step_upd = str(self.t_schedule).split('_')
            t_val, step_upd = float(t_val), int(step_upd)
            if step >= step_upd:
                min_step = int(self.num_train_timesteps * t_val)
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        else:
            raise NotImplementedError(f"{self.t_schedule} is not support.")
        return t

    def set_text_embeddings(self, prompt, negative_prompt, do_classifier_free_guidance):
        if self.text_embeddings is not None:
            return

        # encode text prompt
        text_embeddings, text_embeddings_uncond, text_embeddings_cond = \
            self.encode_prompt(prompt, self.device, do_classifier_free_guidance, negative_prompt=negative_prompt)

        # set pretrained model text embedding
        text_embeddings_uncond, text_embeddings_cond = text_embeddings.chunk(2)
        self.text_embedd_uncond, self.text_embedd_cond = text_embeddings_uncond, text_embeddings_cond
        text_embeddings_unconds = text_embeddings_uncond.repeat_interleave(self.vsd_n_particle, dim=0)
        text_embeddings_conds = text_embeddings_cond.repeat_interleave(self.vsd_n_particle, dim=0)
        text_embeddings = torch.cat([text_embeddings_unconds, text_embeddings_conds])
        self.text_embeddings = text_embeddings

        # set phi model text embedding
        self.text_embeddings_phi = text_embeddings_cond.repeat_interleave(self.phi_n_particle, dim=0)

    def x_augment(self, x: torch.Tensor, img_size: int = 512):
        augment_compose = transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.5, p=0.7),
            transforms.RandomCrop(size=(img_size, img_size), pad_if_needed=True, padding_mode='reflect')
        ])
        return augment_compose(x)

    def variational_score_distillation(self,
                                       pred_rgb: torch.Tensor,
                                       step: int,
                                       prompt: Union[List, str],
                                       negative_prompt: Union[List, str] = None,
                                       grad_scale: float = 1.0,
                                       enhance_particle: bool = False,
                                       im_size: int = 512,
                                       as_latent: bool = False):
        bz = pred_rgb.shape[0]

        # data enhancement for the input particles
        pred_rgb = self.x_augment(pred_rgb, im_size) if enhance_particle else pred_rgb

        # interp to 512x512 to be fed into vae.
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_ = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latent_list = [self.encode2latent(pred_rgb_[i].unsqueeze(0)) for i in range(bz)]
            latents = torch.cat(latent_list, dim=0)
            latents = latents.to(self.device)

        # random sample n_particle_vsd particles from latents
        latents_vsd = latents[torch.randperm(bz)[:self.vsd_n_particle]]

        # encode input prompt
        do_classifier_free_guidance = True
        self.set_text_embeddings(prompt, negative_prompt, do_classifier_free_guidance)
        text_embeddings = self.text_embeddings

        # timestep a.k.a noise level
        self.t = self.schedule_timestep(step)

        # predict the noise residual with unet, stop gradient
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents_vsd)
            latents_noisy = self.scheduler.add_noise(latents_vsd, noise, self.t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2) if do_classifier_free_guidance else latents_noisy
            # pretrained noise prediction network
            noise_pred_pretrain = self.unet(
                latent_model_input, self.t,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs={'scale': 0.0} if self.phi_single else {}
            ).sample

            # use conditional text embeddings in phi_model
            _, text_embeddings_cond = text_embeddings.chunk(2)
            # estimated noise prediction network
            noise_pred_est = self.unet_phi(
                latents_noisy, self.t,
                encoder_hidden_states=text_embeddings_cond,
                cross_attention_kwargs=self.lora_cross_attention_kwargs
            ).sample

        # get pretrained score
        noise_pred_pretrain = self.get_noise_map(noise_pred_pretrain, self.guidance_scale, use_cfg=True)
        # get estimated score
        noise_pred_est = self.get_noise_map(noise_pred_est, self.guidance_scale_lora, use_cfg=False)

        # w(t), sigma_t^2
        w = (1 - self.alphas[self.t]).to(pred_rgb.dtype)
        grad = grad_scale * w * (noise_pred_pretrain - noise_pred_est.detach())
        grad = torch.nan_to_num(grad)

        # grad clipping for stable training
        if self.grad_clip_val is not None and self.grad_clip_val > 0:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # re-parameterization trick:
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents_vsd - grad).detach()
        loss_vpsd = 0.5 * F.mse_loss(latents_vsd, target, reduction="sum")

        return loss_vpsd, grad.norm(), latents, self.t
