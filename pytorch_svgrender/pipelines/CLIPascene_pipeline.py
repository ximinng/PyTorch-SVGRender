import os
import shutil
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from PIL import Image
from pytorch_svgrender.libs.engine import ModelState
from pytorch_svgrender.painter.clipascene import Painter, PainterOptimizer, Loss
from pytorch_svgrender.painter.clipascene.scripts_utils import get_ratios_dict
from pytorch_svgrender.painter.clipascene.sketch_utils import plot_attn, get_mask_u2net, fix_image_scale, \
    log_best_normalised_sketch, inference_sketch
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

    def painterly_rendering(self, im_name):
        self.run_background(im_name)
        self.run_foreground(im_name)

    def run_background(self, im_name):
        print("Start background")
        layers = [int(l) for l in self.args.x.background.layers.split(",")]
        divs = [float(d) for d in self.args.x.background.divs.split(",")]
        self.args.x.resize_obj = 0
        self.args.x.num_iter = self.args.x.background.fidelity.num_iter
        for l in layers:
            if not os.path.exists(self.result_path / "runs" / f"background_l{l}_{im_name}_mask " / "points_mlp.pt"):
                self.generate_fidelity_levels(im_name, l, "background")
        for l, div in zip(layers, divs):
            self.run_ratio(im_name, l, "background", div)

    def run_foreground(self, im_name):
        print("Start foreground")
        layers = [int(l) for l in self.args.x.foreground.layers.split(",")]
        divs = [float(d) for d in self.args.x.foreground.divs.split(",")]
        self.args.x.resize_obj = 1
        for l in layers:
            if not os.path.exists(self.result_path / "runs" / f"object_l{l}_{im_name}" / "points_mlp.pt"):
                self.args.x.num_iter = self.args.x.foreground.num_iter_ge_8 if l >= 8 else self.args.x.foreground.num_iter_lt_8
                self.generate_fidelity_levels(im_name, l, "object")
        for l, div in zip(layers, divs):
            self.run_ratio(im_name, l, "object", div)

    def generate_fidelity_levels(self, im_name, layer_opt, object_or_background):
        print(f"Start fidelity levels for {object_or_background} layer {layer_opt}")
        path_to_input_images = Path("./data")
        output_pref = self.result_path / "runs"
        self.args.x.gradnorm = 0
        if object_or_background == "background":
            folder_ = path_to_input_images / "background"
            im_filename = f"{im_name}_mask.png"
            self.args.x.mask_object = 0
        elif object_or_background == "object":
            folder_ = path_to_input_images / "scene"
            im_filename = f"{im_name}.png"
            if layer_opt != 4:
                self.args.x.gradnorm = 1
            self.args.x.mask_object = 1
        # set the weights for each layer
        clip_conv_layer_weights_int = [0 for _ in range(12)]
        if object_or_background == "object":
            # we combine two layers if we train on objects
            clip_conv_layer_weights_int[4] = 0.5
        clip_conv_layer_weights_int[layer_opt] = 1
        clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
        self.args.x.clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)
        self.args.x.test_name = f"{object_or_background}_l{layer_opt}_{os.path.splitext(im_filename)[0]}"
        self.args.x.num_paths = 64
        self.args.x.num_sketches = 2
        self.args.x.width_optim = 0
        self.args.x.width_loss_weight = 0
        self.args.x.path_svg = "none"
        self.args.x.mlp_width_weights_path = "none"
        self.args.x.mlp_points_weights_path = "none"
        self.args.x.load_points_opt_weights = 0
        self.args.x.width_weights_lst = ""
        self.args.x.ratio_loss = 0
        self.args.x.eval_interval = 50
        self.args.x.min_eval_iter = 400
        self.run_sketch(target_file=folder_ / im_filename, output_pref=output_pref)

    def run_ratio(self, im_name, layer_opt, object_or_background, min_div):
        print(f"Start ratio for {object_or_background} layer {layer_opt}")
        path_to_files = Path("./data")  # where the input images are located
        output_pref = self.result_path / "runs"  # path to output the results
        path_res_pref = self.result_path / "runs"  # path to take semantic trained models from
        if object_or_background == "background":
            folder_ = path_to_files / "background"
            filename = f"{im_name}_mask.png"
        elif object_or_background == "object":
            folder_ = path_to_files / "scene"
            filename = f"{im_name}.png"
        file_ = folder_ / filename
        res_filename = f"{object_or_background}_l{layer_opt}_{os.path.splitext(filename)[0]}"

        clip_conv_layer_weights_int = [0 for k in range(12)]
        if object_or_background == "object":
            clip_conv_layer_weights_int[4] = 0.5
        clip_conv_layer_weights_int[layer_opt] = 1
        clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
        self.args.x.clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

        # load the semantic MLP and its input
        path_res = path_res_pref / res_filename
        self.args.x.path_svg = path_res / "init.svg"
        self.args.x.mlp_points_weights_path = path_res / "points_mlp.pt"
        assert os.path.exists(self.args.x.mlp_points_weights_path)

        ratios_str = get_ratios_dict(path_res_pref, folder_name_l=res_filename,
                                     layer=layer_opt, im_name=im_name,
                                     object_or_background=object_or_background,
                                     step_size_l=min_div)
        self.args.x.width_weights_lst = ratios_str
        self.args.x.num_paths = 64
        self.args.x.num_iter = 401
        self.args.x.num_sketches = 2
        self.args.x.width_optim = 1
        self.args.x.width_loss_weight = 1
        self.args.x.gradnorm = 1
        self.args.x.eval_interval = 100
        self.args.x.min_eval_iter = 100
        ratios = [float(item) for item in ratios_str.split(',')]
        for i, ratio in enumerate(ratios):
            test_name_pref = f"l{layer_opt}_{os.path.splitext(os.path.basename(file_))[0]}_{min_div}"
            test_name = f"ratio{ratio}_{test_name_pref}"
            self.args.x.ratio_loss = ratio
            self.args.x.test_name = test_name
            if not os.path.exists(output_pref / test_name / "width_mlp.pt"):
                if i == 0:
                    # in this case we use the semantic mlp (first row) and we don't want its optimizer
                    self.args.x.mlp_width_weights_path = "none"
                    self.args.x.load_points_opt_weights = 0
                else:
                    self.args.x.mlp_width_weights_path = output_pref / f"ratio{ratios[i - 1]}_{test_name_pref}" / "width_mlp.pt"
                    assert os.path.exists(self.args.x.mlp_width_weights_path)

                    self.args.x.mlp_points_weights_path = output_pref / f"ratio{ratios[i - 1]}_{test_name_pref}" / "points_mlp.pt"
                    assert os.path.exists(self.args.x.mlp_points_weights_path)

                    self.args.x.load_points_opt_weights = 1
                self.run_sketch(target_file=file_, output_pref=output_pref)

    def run_sketch(self, target_file, output_pref):
        print("Start sketching")
        output_dir = output_pref / self.args.x.test_name
        if self.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "mlps").mkdir(parents=True, exist_ok=True)
        losses_eval_sum = {}
        losses_best_normalised = {}
        # run the sketching

        seeds = list(range(0, self.args.x.num_sketches * 1000, 1000))
        for j in range(self.args.x.num_sketches):
            seed = seeds[j]
            wandb_name = f"{self.args.x.test_name}_seed{seed}"
            final_config = vars(self.args.x)
            configs_to_save = self.paint(target_file, output_dir / wandb_name)
            for k in configs_to_save.keys():
                final_config[k] = configs_to_save[k]
            np.save(output_dir / wandb_name / "config.npy", final_config)
            try:
                config = np.load(output_dir / wandb_name / "config.npy", allow_pickle=True)[()]
            except Exception as e:
                print(e)
            if self.args.x.width_optim:
                losses_best_normalised[wandb_name] = config["best_normalised_loss"]

            loss_eval = np.array(config['loss_eval'])
            inds = np.argsort(loss_eval)
            losses_eval_sum[wandb_name] = loss_eval[inds][0]

        # save the mlps for the best normalised loss
        if self.args.x.width_optim:
            sorted_final_n = dict(sorted(losses_best_normalised.items(), key=lambda item: item[1]))
            winning_trial = list(sorted_final_n.keys())[0]
            copyfile(output_dir / winning_trial / "svg_logs" / "init_svg.svg",
                     output_dir / "init.svg")
            copyfile(output_dir / winning_trial / "best_iter.svg",
                     output_dir / f"{winning_trial}_best.svg")
            copyfile(output_dir / winning_trial / "points_mlp.pt",
                     output_dir / "points_mlp.pt")
            copyfile(output_dir / winning_trial / "width_mlp.pt",
                     output_dir / "width_mlp.pt")
            for folder_name in list(sorted_final_n.keys()):
                shutil.rmtree(output_dir / folder_name / "mlps")

        # in this case it's a baseline run to produce the first row
        else:
            sorted_final = dict(sorted(losses_eval_sum.items(), key=lambda item: item[1]))
            winning_trial = list(sorted_final.keys())[0]
            copyfile(output_dir / winning_trial / "svg_logs" / "init_svg.svg",
                     output_dir / "init.svg")
            copyfile(output_dir / winning_trial / "best_iter.svg",
                     output_dir / f"{winning_trial}_best.svg")
            copyfile(output_dir / winning_trial / "points_mlp.pt",
                     output_dir / "points_mlp.pt")
            copyfile(output_dir / winning_trial / "mask.png",
                     output_dir / "mask.png")
            if os.path.exists(output_dir / winning_trial / "resize_params.npy"):
                copyfile(output_dir / winning_trial / "resize_params.npy",
                         output_dir / "resize_params.npy")

    def paint(self, target, output_dir):
        print("Start painting")
        png_log_dir = output_dir / "png_logs"
        svg_log_dir = output_dir / "svg_logs"
        mlps_logs_dir = output_dir / "mlps"
        if self.accelerator.is_main_process:
            png_log_dir.mkdir(parents=True, exist_ok=True)
            svg_log_dir.mkdir(parents=True, exist_ok=True)
            mlps_logs_dir.mkdir(parents=True, exist_ok=True)
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
        configs_to_save = {"loss_eval": []}
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
            init_sketches = renderer.get_image("init").to(self.device)
            renderer.save_svg(self.result_path, "init")

        total_step = self.args.x.num_iter
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

                if step % self.args.x.save_interval == 0:
                    plot_batch(inputs,
                               sketches,
                               self.step,
                               save_path=png_log_dir.as_posix(),
                               name=f"iter{step}")
                    renderer.save_svg(svg_log_dir.as_posix(), f"svg_iter{step}")

                if step % self.args.x.eval_interval == 0 and step >= self.args.x.min_eval_iter:
                    if self.args.x.width_optim:
                        if self.args.x.mlp_train and self.args.x.optimize_points:
                            torch.save({
                                'model_state_dict': renderer.get_mlp().state_dict(),
                                'optimizer_state_dict': optimizer.get_points_optim().state_dict(),
                            }, output_dir / "mlps" / f"points_mlp{step}.pt")
                        torch.save({
                            'model_state_dict': renderer.get_width_mlp().state_dict(),
                            'optimizer_state_dict': optimizer.get_width_optim().state_dict(),
                        }, output_dir / "mlps" / f"width_mlp{step}.pt")

                    with torch.no_grad():
                        losses_dict_weighted_eval, losses_dict_norm_eval, losses_dict_original_eval = loss_func(
                            sketches,
                            inputs,
                            step,
                            renderer.get_widths(),
                            renderer=renderer,
                            mode="eval",
                            width_opt=renderer.width_optim)
                        loss_eval = sum(list(losses_dict_weighted_eval.values()))
                        configs_to_save["loss_eval"].append(loss_eval.item())
                        if "num_strokes" not in configs_to_save.keys():
                            configs_to_save["num_strokes"] = []
                        configs_to_save["num_strokes"].append(renderer.get_strokes_count())
                        for k in losses_dict_norm_eval.keys():
                            original_name, gradnorm_name, final_name = k + "_original_eval", k + "_gradnorm_eval", k + "_final_eval"
                            if original_name not in configs_to_save.keys():
                                configs_to_save[original_name] = []
                            if gradnorm_name not in configs_to_save.keys():
                                configs_to_save[gradnorm_name] = []
                            if final_name not in configs_to_save.keys():
                                configs_to_save[final_name] = []

                            configs_to_save[original_name].append(losses_dict_original_eval[k].item())
                            configs_to_save[gradnorm_name].append(losses_dict_norm_eval[k].item())
                            if k in losses_dict_weighted_eval.keys():
                                configs_to_save[final_name].append(losses_dict_weighted_eval[k].item())

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

                                if self.args.x.mlp_train and self.args.x.optimize_points and not self.args.x.width_optim:
                                    torch.save({
                                        'model_state_dict': renderer.get_mlp().state_dict(),
                                        'optimizer_state_dict': optimizer.get_points_optim().state_dict(),
                                    }, output_dir / "points_mlp.pt")

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

        if self.args.x.width_optim:
            log_best_normalised_sketch(configs_to_save, output_dir, self.args.x.eval_interval,
                                       self.args.x.min_eval_iter)
        inference_sketch(self.args.x, output_dir, device=self.device)
        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (output_dir / f"clipascene_sketch.mp4").as_posix()
            ])
        return configs_to_save

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
