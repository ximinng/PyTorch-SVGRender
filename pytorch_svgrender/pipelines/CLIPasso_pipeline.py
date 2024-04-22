# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
from PIL import Image

import torch
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets.folder import is_image_file

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.painter.clipasso import Painter, PainterOptimizer, Loss
from pytorch_svgrender.painter.clipasso.sketch_utils import plot_attn, get_mask_u2net, fix_image_scale
from pytorch_svgrender.plt import plot_img, plot_couple, plot_img_title


class CLIPassoPipeline(ModelState):

    def __init__(self, args):
        logdir_ = f"sd{args.seed}" \
                  f"-im{args.x.image_size}" \
                  f"{'-mask' if args.x.mask_object else ''}" \
                  f"{'-XDoG' if args.x.xdog_intersec else ''}" \
                  f"-P{args.x.num_paths}W{args.x.width}{'OP' if args.x.force_sparse else 'BL'}" \
                  f"-tau{args.x.softmax_temp}"
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

    def painterly_rendering(self, image_path):
        loss_func = Loss(self.x_cfg, self.device)

        # preprocess input image
        inputs, mask = self.get_target(image_path,
                                       self.x_cfg.image_size,
                                       self.result_path,
                                       self.x_cfg.u2net_path,
                                       self.x_cfg.mask_object,
                                       self.x_cfg.fix_scale,
                                       self.device)
        plot_img(inputs, self.result_path, fname="input")

        # init renderer
        renderer = self.load_renderer(inputs, mask)
        img = renderer.init_image(stage=0)
        self.print("init_image shape: ", img.shape)
        plot_img(img, self.result_path, fname="init_img")

        # init optimizer
        optimizer = PainterOptimizer(renderer,
                                     self.x_cfg.num_iter,
                                     self.x_cfg.lr,
                                     self.x_cfg.force_sparse, self.x_cfg.color_lr)
        optimizer.init_optimizers()

        best_loss, best_fc_loss = 100, 100
        min_delta = 1e-5
        total_step = self.x_cfg.num_iter

        with tqdm(initial=self.step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_step:
                sketches = renderer.get_image().to(self.device)

                if self.make_video and (self.step % self.args.framefreq == 0 or self.step == total_step - 1):
                    plot_img(sketches, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                losses_dict = loss_func(sketches,
                                        inputs.detach(),
                                        renderer.get_color_parameters(),
                                        renderer,
                                        self.step,
                                        optimizer)
                loss = sum(list(losses_dict.values()))

                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                if self.x_cfg.lr_schedule:
                    optimizer.update_lr()

                pbar.set_description(f"L_train: {loss.item():.5f}")

                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    plot_couple(inputs,
                                sketches,
                                self.step,
                                output_dir=self.png_logs_dir.as_posix(),
                                fname=f"iter{self.step}")
                    renderer.save_svg(self.svg_logs_dir.as_posix(), f"svg_iter{self.step}")

                if self.step % self.args.eval_step == 0 and self.accelerator.is_main_process:
                    with torch.no_grad():
                        losses_dict_eval = loss_func(
                            sketches,
                            inputs,
                            renderer.get_color_parameters(),
                            renderer.get_point_parameters(),
                            self.step,
                            optimizer,
                            mode="eval"
                        )
                        loss_eval = sum(list(losses_dict_eval.values()))

                        cur_delta = loss_eval.item() - best_loss
                        if abs(cur_delta) > min_delta and cur_delta < 0:
                            best_loss = loss_eval.item()
                            best_iter = self.step
                            plot_couple(inputs,
                                        sketches,
                                        best_iter,
                                        output_dir=self.result_path.as_posix(),
                                        fname="best_iter")
                            renderer.save_svg(self.result_path.as_posix(), "best_iter")

                if self.step == 0 and self.x_cfg.attention_init and self.accelerator.is_main_process:
                    plot_attn(renderer.get_attn(),
                              renderer.get_thresh(),
                              inputs,
                              renderer.get_inds(),
                              (self.result_path / "attention_map.png").as_posix(),
                              self.x_cfg.saliency_model)

                self.step += 1
                pbar.update(1)

        # log final results
        renderer.save_svg(self.result_path.as_posix(), "final_render")
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
                (self.result_path / "clipasso_rendering.mp4").as_posix()
            ])

        self.close(msg="painterly rendering complete.")

    def load_renderer(self, target_im=None, mask=None):
        renderer = Painter(method_cfg=self.x_cfg,
                           diffvg_cfg=self.args.diffvg,
                           num_strokes=self.x_cfg.num_paths,
                           canvas_size=self.x_cfg.image_size,
                           device=self.device,
                           target_im=target_im,
                           mask=mask)
        return renderer

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

        # U^2 net mask
        masked_im, mask = get_mask_u2net(target, output_dir, u2net_path, device)
        if mask_object:
            target = masked_im

        if fix_scale:
            target = fix_image_scale(target)

        transforms_ = []
        if target.size[0] != target.size[1]:
            transforms_.append(
                transforms.Resize((image_size, image_size),
                                  interpolation=InterpolationMode.BICUBIC)
            )
        else:
            transforms_.append(transforms.Resize(image_size,
                                                 interpolation=InterpolationMode.BICUBIC))
            transforms_.append(transforms.CenterCrop(image_size))

        transforms_.append(transforms.ToTensor())
        data_transforms = transforms.Compose(transforms_)
        target_ = data_transforms(target).unsqueeze(0).to(self.device)
        return target_, mask
