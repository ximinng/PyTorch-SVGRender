import pathlib
import random

import omegaconf
import pydiffvg
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

from pytorch_svgrender.diffvg_warp import DiffVGState
from pytorch_svgrender.libs.modules.edge_map.DoG import XDoG
from .grad_cam import gradCAM
from . import modified_clip as clip


class Painter(DiffVGState):

    def __init__(
            self,
            method_cfg: omegaconf.DictConfig,
            diffvg_cfg: omegaconf.DictConfig,
            num_strokes: int = 4,
            canvas_size: int = 224,
            device=None,
            target_im=None,
            mask=None
    ):
        super(Painter, self).__init__(device, print_timing=diffvg_cfg.print_timing,
                                      canvas_width=canvas_size, canvas_height=canvas_size)

        self.args = method_cfg
        self.num_paths = num_strokes
        self.num_segments = method_cfg.num_segments
        self.width = method_cfg.width
        self.control_points_per_seg = method_cfg.control_points_per_seg
        self.opacity_optim = method_cfg.force_sparse
        self.num_stages = method_cfg.num_stages
        self.noise_thresh = method_cfg.noise_thresh
        self.softmax_temp = method_cfg.softmax_temp

        self.color_vars_threshold = method_cfg.color_vars_threshold

        self.path_svg = method_cfg.path_svg
        self.strokes_per_stage = self.num_paths
        self.optimize_flag = []

        # attention related for strokes initialisation
        self.attention_init = method_cfg.attention_init
        self.saliency_model = method_cfg.saliency_model
        self.xdog_intersec = method_cfg.xdog_intersec
        self.mask_object = method_cfg.mask_object_attention

        self.text_target = method_cfg.text_target  # for clip gradients
        self.saliency_clip_model = method_cfg.saliency_clip_model
        self.image2clip_input = self.clip_preprocess(target_im)
        self.mask = mask
        self.attention_map = self.set_attention_map() if self.attention_init else None

        self.thresh = self.set_attention_threshold_map() if self.attention_init else None
        self.strokes_counter = 0  # counts the number of calls to "get_path"
        self.epoch = 0
        self.final_epoch = method_cfg.num_iter - 1

    def init_image(self, stage=0):
        if stage > 0:
            # Noting: if multi stages training than add new strokes on existing ones
            # don't optimize on previous strokes
            self.optimize_flag = [False for i in range(len(self.shapes))]
            for i in range(self.strokes_per_stage):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                                 fill_color=None,
                                                 stroke_color=stroke_color)
                self.shape_groups.append(path_group)
                self.optimize_flag.append(True)
        else:
            num_paths_exists = 0
            if self.path_svg is not None and pathlib.Path(self.path_svg).exists():
                print(f"-> init svg from `{self.path_svg}` ...")

                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = self.load_svg(self.path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_paths_exists = len(self.shapes)

            for i in range(num_paths_exists, self.num_paths):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                                 fill_color=None,
                                                 stroke_color=stroke_color)
                self.shape_groups.append(path_group)
            self.optimize_flag = [True for i in range(len(self.shapes))]

        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (
                1 - img[:, :, 3:4])
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW

        return img

    def get_image(self):
        img = self.render_warp()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_path(self):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        p0 = self.inds_normalised[self.strokes_counter] if self.attention_init else (random.random(), random.random())
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height

        path = pydiffvg.Path(num_control_points=self.num_control_points,
                             points=points,
                             stroke_width=torch.tensor(self.width),
                             is_closed=False)
        self.strokes_counter += 1
        return path

    def render_warp(self):
        if self.opacity_optim:
            for group in self.shape_groups:
                group.stroke_color.data[:3].clamp_(0., 0.)  # to force black stroke
                group.stroke_color.data[-1].clamp_(0., 1.)  # opacity
                # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()

        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups
        )
        img = _render(self.canvas_width,  # width
                      self.canvas_height,  # height
                      2,  # num_samples_x
                      2,  # num_samples_y
                      0,  # seed
                      None,
                      *scene_args)
        return img

    def set_point_parameters(self):
        self.point_vars = []
        # storkes' location optimization
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.requires_grad = True
                self.point_vars.append(path.points)

    def get_point_parameters(self):
        return self.point_vars

    def set_color_parameters(self):
        # for storkes' color optimization (opacity)
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if self.optimize_flag[i]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)

    def get_color_parameters(self):
        return self.color_vars

    def save_svg(self, output_dir: str, name: str):
        pydiffvg.save_svg(f'{output_dir}/{name}.svg',
                          self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)

    def clip_preprocess(self, target_im):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([
            preprocess.transforms[-1],
        ])
        return data_transforms(target_im).to(self.device)

    def clip_attn(self):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)

        if "RN" in self.saliency_clip_model:
            text_input = clip.tokenize([self.text_target]).to(self.device)
            saliency_layer = "layer4"
            attn_map = gradCAM(
                model.visual,
                self.image2clip_input,
                model.encode_text(text_input).float(),
                getattr(model.visual, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        else:  # ViT
            attn_map = interpret(self.image2clip_input, model, self.device)

        del model
        return attn_map

    def set_attention_map(self):
        assert self.saliency_model in ["clip"]
        if self.saliency_model == "clip":
            return self.clip_attn()

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()

    def set_inds_clip(self):
        attn_map = (self.attention_map - self.attention_map.min()) / \
                   (self.attention_map.max() - self.attention_map.min())
        if self.xdog_intersec:
            xdog = XDoG(k=10)
            im_xdog = xdog(self.image2clip_input[0].permute(1, 2, 0).cpu().numpy())
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map

        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)

        k = self.num_stages * self.num_paths
        self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False,
                                     p=attn_map_soft.flatten())
        self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return attn_map_soft

    def set_attention_threshold_map(self):
        assert self.saliency_model in ["clip"]
        if self.saliency_model == "clip":
            return self.set_inds_clip()

    def get_attn(self):
        return self.attention_map

    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds

    def get_mask(self):
        return self.mask


class PainterOptimizer:

    def __init__(self, renderer: Painter, num_iter: int, points_lr: float, force_sparse: bool, color_lr: float):
        self.renderer = renderer
        self.num_iter = num_iter
        self.points_lr = points_lr
        self.color_lr = color_lr
        self.optim_color = force_sparse

        self.points_optimizer, self.color_optimizer = None, None
        self.scheduler = None

    def init_optimizers(self):
        # optimizers
        self.renderer.set_point_parameters()
        self.points_optimizer = torch.optim.Adam(self.renderer.get_point_parameters(), lr=self.points_lr)
        if self.optim_color:
            self.renderer.set_color_parameters()
            self.color_optimizer = torch.optim.Adam(self.renderer.get_color_parameters(), lr=self.color_lr)
        # lr schedule
        lr_lambda_fn = LinearDecayLR(self.num_iter, 0.4)
        self.scheduler = LambdaLR(self.points_optimizer, lr_lambda=lr_lambda_fn, last_epoch=-1)

    def update_lr(self):
        self.scheduler.step()

    def zero_grad_(self):
        self.points_optimizer.zero_grad()
        if self.optim_color:
            self.color_optimizer.zero_grad()

    def step_(self):
        self.points_optimizer.step()
        if self.optim_color:
            self.color_optimizer.step()

    def get_lr(self):
        return self.points_optimizer.param_groups[0]['lr']


class LinearDecayLR:

    def __init__(self, decay_every, decay_ratio):
        self.decay_every = decay_every
        self.decay_ratio = decay_ratio

    def __call__(self, n):
        decay_time = n // self.decay_every
        decay_step = n % self.decay_every
        lr_s = self.decay_ratio ** decay_time
        lr_e = self.decay_ratio ** (decay_time + 1)
        r = decay_step / self.decay_every
        lr = lr_s * (1 - r) + lr_e * r
        return lr


def interpret(image, clip_model, device):
    # virtual forward to get attention map
    images = image.repeat(1, 1, 1, 1)
    _ = clip_model.encode_image(images)  # ensure `attn_probs` in attention is not empty
    clip_model.zero_grad()

    image_attn_blocks = list(dict(clip_model.visual.transformer.resblocks.named_children()).values())
    # create R to store attention map
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)

    cams = []
    for i, blk in enumerate(image_attn_blocks):  # 12 attention blocks
        cam = blk.attn_probs.detach()  # attn_probs shape: [12, 50, 50]
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1)  # mean of the 12 something
        cams.append(cam)
        R = R + torch.bmm(cam, R)

    cams_avg = torch.cat(cams)  # [12, 50, 50]
    cams_avg = cams_avg[:, 0, 1:]  # [12, 49]
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)  # [1, 49]
    image_relevance = image_relevance.reshape(1, 1, 7, 7)  # [1, 1, 7, 7]
    # interpolate: [1, 1, 7, 7] -> [1, 3, 224, 224]
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    # normalize the tensor to [0, 1]
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance
