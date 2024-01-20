import os
from pathlib import Path

import torch
from PIL import Image
from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.painter.clipascene import Painter, PainterOptimizer, Loss
from pytorch_svgrender.painter.clipascene.sketch_utils import plot_attn, get_mask_u2net, fix_image_scale
from pytorch_svgrender.plt import log_input, plot_batch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm


class CLIPascenePipeline(ModelState):
    def __init__(self, args):
        logdir_ = f"sd{args.seed}" \
                  f"-im{args.target}" \
                  f"-P{args.x.num_paths}W{args.x.width}"
        super().__init__(args, log_path_suffix=logdir_)
        self.path_to_input_images = Path("./data")

    def painterly_rendering(self, im_name):
        self.run_background(im_name)
        self.run_foreground(im_name)

    def run_background(self, im_name):
        print("=====Start background=====")
        self.args.x.resize_obj = 0

        folder_ = self.path_to_input_images / "background"
        im_filename = f"{im_name}_mask.png"
        target_file = folder_ / im_filename
        self.args.x.mask_object = 0

        clip_conv_layer_weights_int = [0 for _ in range(12)]
        clip_conv_layer_weights_int[self.args.x.background_layer] = 1
        clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
        self.args.x.clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

        output_dir = self.result_path / f"background_l{self.args.x.background_layer}_{os.path.splitext(im_filename)[0]}"
        if self.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.paint(target_file, output_dir, self.args.x.background_num_iter)

    def run_foreground(self, im_name):
        print("=====Start foreground=====")
        self.args.x.resize_obj = 1

        folder_ = self.path_to_input_images / "scene"
        im_filename = f"{im_name}.png"
        target_file = folder_ / im_filename
        if self.args.x.foreground_layer != 4:
            self.args.x.gradnorm = 1
        self.args.x.mask_object = 1

        clip_conv_layer_weights_int = [0 for _ in range(12)]
        clip_conv_layer_weights_int[4] = 0.5
        clip_conv_layer_weights_int[self.args.x.foreground_layer] = 1
        clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
        self.args.x.clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

        output_dir = self.result_path / f"object_l{self.args.x.foreground_layer}_{os.path.splitext(im_filename)[0]}"
        if self.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.paint(target_file, output_dir, self.args.x.foreground_num_iter)

    def paint(self, target, output_dir, num_iter):
        png_log_dir = output_dir / "png_logs"
        svg_log_dir = output_dir / "svg_logs"
        if self.accelerator.is_main_process:
            png_log_dir.mkdir(parents=True, exist_ok=True)
            svg_log_dir.mkdir(parents=True, exist_ok=True)
        # make video log
        self.make_video = self.args.mv
        if self.make_video:
            self.frame_idx = 0
            self.frame_log_dir = output_dir / "frame_logs"
            self.frame_log_dir.mkdir(parents=True, exist_ok=True)
        # preprocess input image
        inputs, mask = self.get_target(target,
                                       self.args.x.image_size,
                                       output_dir,
                                       self.args.x.u2net_path,
                                       self.args.x.mask_object,
                                       self.args.x.fix_scale,
                                       self.device)
        log_input(inputs, output_dir)
        loss_func = Loss(self.x_cfg, mask, self.device)
        # init renderer
        renderer = self.load_renderer(inputs, mask)

        # init optimizer
        optimizer = PainterOptimizer(self.x_cfg, renderer)
        best_loss, best_fc_loss, best_num_strokes = 100, 100, self.args.x.num_paths
        best_iter, best_iter_fc = 0, 0
        min_delta = 1e-7
        renderer.set_random_noise(0)
        renderer.init_image(stage=0)
        renderer.save_svg(svg_log_dir, "init_svg")
        optimizer.init_optimizers()

        if self.args.x.switch_loss:
            # start with width optim and than switch every switch_loss iterations
            renderer.turn_off_points_optim()
            optimizer.turn_off_points_optim()

        with torch.no_grad():
            renderer.get_image("init").to(self.device)
            renderer.save_svg(self.result_path, "init")

        total_step = num_iter
        step = 0
        with tqdm(initial=step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            while step < total_step:
                optimizer.zero_grad_()
                sketches = renderer.get_image().to(self.device)
                if self.make_video and (step % self.args.framefreq == 0 or step == total_step - 1):
                    log_input(sketches, self.frame_log_dir, output_prefix=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                losses_dict_weighted, _, _ = loss_func(sketches, inputs.detach(), step,
                                                       renderer.get_widths(), renderer,
                                                       optimizer, mode="train",
                                                       width_opt=renderer.width_optim)
                loss = sum(list(losses_dict_weighted.values()))
                loss.backward()
                optimizer.step_()

                if step % self.args.x.save_step == 0:
                    plot_batch(inputs,
                               sketches,
                               self.step,
                               save_path=png_log_dir.as_posix(),
                               name=f"iter{step}")
                    renderer.save_svg(svg_log_dir.as_posix(), f"svg_iter{step}")

                if step % self.args.x.eval_step == 0 and step >= self.args.x.min_eval_iter:

                    with torch.no_grad():
                        losses_dict_weighted_eval, _, _ = loss_func(
                            sketches,
                            inputs,
                            step,
                            renderer.get_widths(),
                            renderer=renderer,
                            mode="eval",
                            width_opt=renderer.width_optim)
                        loss_eval = sum(list(losses_dict_weighted_eval.values()))

                        cur_delta = loss_eval.item() - best_loss
                        if abs(cur_delta) > min_delta:
                            if cur_delta < 0:
                                best_loss = loss_eval.item()
                                best_iter = step
                                plot_batch(inputs,
                                           sketches,
                                           best_iter,
                                           save_path=output_dir.as_posix(),
                                           name="best_iter")
                                renderer.save_svg(output_dir.as_posix(), "best_iter")

                if step == 0 and self.x_cfg.attention_init and self.accelerator.is_main_process:
                    plot_attn(renderer.get_attn(),
                              renderer.get_thresh(),
                              inputs,
                              renderer.get_inds(),
                              (output_dir / "attention_map.png").as_posix(),
                              self.x_cfg.saliency_model)

                if self.args.x.switch_loss:
                    if step > 0 and step % self.args.x.switch_loss == 0:
                        renderer.switch_opt()
                        optimizer.switch_opt()

                step += 1
                pbar.update(1)

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (output_dir / f"clipascene_sketch.mp4").as_posix()
            ])

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
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)
            )
        else:
            transforms_.append(transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC))
            transforms_.append(transforms.CenterCrop(image_size))

        transforms_.append(transforms.ToTensor())
        data_transforms = transforms.Compose(transforms_)
        target_ = data_transforms(target).unsqueeze(0).to(self.device)
        return target_, mask
