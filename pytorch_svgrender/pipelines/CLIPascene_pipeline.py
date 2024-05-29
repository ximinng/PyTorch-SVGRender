import shutil
from PIL import Image
from pathlib import Path

from tqdm.auto import tqdm
import imageio
import numpy as np
from skimage.transform import resize
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.painter.clipascene import Painter, PainterOptimizer, Loss
from pytorch_svgrender.painter.clipascene.lama_utils import apply_inpaint
from pytorch_svgrender.painter.clipascene.scripts_utils import read_svg
from pytorch_svgrender.painter.clipascene.sketch_utils import plot_attn, get_mask_u2net, fix_image_scale
from pytorch_svgrender.plt import plot_img, plot_couple
from pytorch_svgrender.svgtools import merge_svg_files


class CLIPascenePipeline(ModelState):
    def __init__(self, args):
        logdir_ = f"sd{args.seed}" \
                  f"-im{args.x.image_size}" \
                  f"-P{args.x.num_paths}W{args.x.width}"
        super().__init__(args, log_path_suffix=logdir_)

    def painterly_rendering(self, image_path):
        foreground_target, background_target = self.preprocess_image(image_path)
        background_output_dir = self.run_background(background_target)
        foreground_output_dir = self.run_foreground(foreground_target)
        self.combine(background_output_dir, foreground_output_dir, self.device)
        self.close(msg="painterly rendering complete.")

    def preprocess_image(self, image_path):
        image_path = Path(image_path)
        scene_path = self.result_path / "scene"
        background_path = self.result_path / "background"
        if self.accelerator.is_main_process:
            scene_path.mkdir(parents=True, exist_ok=True)
            background_path.mkdir(parents=True, exist_ok=True)

        im = Image.open(image_path)
        max_size = max(im.size[0], im.size[1])
        scaled_path = scene_path / f"{image_path.stem}.png"
        if max_size > 512:
            im = Image.open(image_path).convert("RGB").resize((512, 512))
            im.save(scaled_path)
        else:
            shutil.copyfile(image_path, scaled_path)

        scaled_img = Image.open(scaled_path)
        mask = get_mask_u2net(scaled_img, scene_path, self.args.x.u2net_path, preprocess=True, device=self.device)
        masked_path = scene_path / f"{image_path.stem}_mask.png"
        imageio.imsave(masked_path, mask.astype(np.uint8) * 255)

        apply_inpaint(scene_path, background_path, self.device)
        return scaled_path, background_path / f"{image_path.stem}_mask.png"

    def run_background(self, target_file):
        print("=====Start background=====")
        self.args.x.resize_obj = 0
        self.args.x.mask_object = 0

        clip_conv_layer_weights_int = [0 for _ in range(12)]
        clip_conv_layer_weights_int[self.args.x.background_layer] = 1
        clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
        self.args.x.clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

        output_dir = self.result_path / "background"
        if self.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.paint(target_file, output_dir, self.args.x.background_num_iter)
        print("=====End background=====")
        return output_dir

    def run_foreground(self, target_file):
        print("=====Start foreground=====")
        self.args.x.resize_obj = 1
        if self.args.x.foreground_layer != 4:
            self.args.x.gradnorm = 1
        self.args.x.mask_object = 1

        clip_conv_layer_weights_int = [0 for _ in range(12)]
        clip_conv_layer_weights_int[4] = 0.5
        clip_conv_layer_weights_int[self.args.x.foreground_layer] = 1
        clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
        self.args.x.clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

        output_dir = self.result_path / "object"
        if self.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.paint(target_file, output_dir, self.args.x.foreground_num_iter)
        print("=====End foreground=====")
        return output_dir

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
                                       self.args.x.resize_obj,
                                       self.args.x.u2net_path,
                                       self.args.x.mask_object,
                                       self.args.x.fix_scale,
                                       self.device)
        plot_img(inputs, output_dir, fname="target")
        loss_func = Loss(self.x_cfg, mask, self.device)
        # init renderer
        renderer = self.load_renderer(inputs, mask)
        # init optimizer
        optimizer = PainterOptimizer(self.x_cfg, renderer)

        # set renderer and optimizer
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

        best_loss, best_fc_loss, best_num_strokes = 100, 100, self.args.x.num_paths
        min_delta = 1e-7
        total_step = num_iter
        step = 0
        with tqdm(initial=step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            while step < total_step:
                optimizer.zero_grad_()
                sketches = renderer.get_image().to(self.device)
                if self.make_video and (step % self.args.framefreq == 0 or step == total_step - 1):
                    plot_img(sketches, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                losses_dict_weighted, _, _ = loss_func(sketches, inputs.detach(), step,
                                                       renderer.get_widths(), renderer,
                                                       optimizer, mode="train",
                                                       width_opt=renderer.width_optim)
                loss = sum(list(losses_dict_weighted.values()))
                loss.backward()
                optimizer.step_()

                if step % self.args.x.save_step == 0:
                    plot_couple(inputs,
                                sketches,
                                self.step,
                                output_dir=png_log_dir.as_posix(),
                                fname=f"iter{step}")
                    renderer.save_svg(svg_log_dir.as_posix(), f"svg_iter{step}")

                if step % self.args.x.eval_step == 0:
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
                                plot_couple(inputs,
                                            sketches,
                                            best_iter,
                                            output_dir=output_dir.as_posix(),
                                            fname="best_iter")
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
                   resize_obj,
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
        masked_im, mask = get_mask_u2net(target, output_dir, u2net_path, resize_obj=resize_obj, device=device)
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

    def combine(self, background_output_dir, foreground_output_dir, device, output_size=448):
        params_path = foreground_output_dir / "resize_params.npy"
        params = None
        if params_path.exists():
            params = np.load(params_path, allow_pickle=True)[()]
        mask_path = foreground_output_dir / "mask.png"
        # mask = imageio.imread(mask_path)
        mask = imageio.v2.imread(mask_path)
        mask = resize(mask, (output_size, output_size), anti_aliasing=False)

        foreground_svg_path = foreground_output_dir / "best_iter.svg"
        raster_o = read_svg(foreground_svg_path, resize_obj=1, params=params, multiply=1.8, device=device)

        background_svg_path = background_output_dir / "best_iter.svg"
        raster_b = read_svg(background_svg_path, resize_obj=0, params=params, multiply=1.8, device=device)

        combine_svg_path = self.result_path / "combined.svg"
        merge_svg_files(foreground_svg_path, background_svg_path, merge_type='simple',
                        output_svg_path=combine_svg_path.as_posix(),
                        out_size=(self.x_cfg.image_size, self.x_cfg.image_size))

        raster_b[mask == 1] = 1
        raster_b[raster_o != 1] = raster_o[raster_o != 1]
        raster_b = torch.from_numpy(raster_b).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        plot_img(raster_b, self.result_path, fname="combined")
