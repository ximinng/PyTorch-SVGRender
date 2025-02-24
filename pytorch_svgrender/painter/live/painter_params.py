# -*- coding: utf-8 -*-
# Author: ximing
# Description: LIVE painter and optimizer
# Copyright (c) 2023, XiMing Xing.
# License: MIT License

import copy
import random

import omegaconf
import cv2
import numpy as np
import pydiffvg
import torch
from torch.optim.lr_scheduler import LambdaLR

from pytorch_svgrender.model_helper import DiffVGState


class Painter(DiffVGState):

    def __init__(
            self,
            target_img: torch.Tensor,
            diffvg_cfg: omegaconf.DictConfig,
            num_segments: int = 4,
            segment_init: str = 'random',
            radius: int = 5,
            canvas_size=240,
            trainable_bg: bool = False,
            stroke: bool = False,
            stroke_width: int = 3,
            device: torch.device = None,
    ):
        super(Painter, self).__init__(device, print_timing=diffvg_cfg.print_timing,
                                      canvas_width=canvas_size, canvas_height=canvas_size)

        self.target_img = target_img

        self.num_segments = num_segments
        self.segment_init = segment_init
        self.radius = radius
        self.train_stroke = stroke
        self.stroke_width = stroke_width

        self.points_vars = []
        self.stroke_width_vars = []
        self.stroke_color_vars = []
        self.color_vars = []

        self.strokes_counter = 0  # counts the number of calls to "get_path"

        # Background
        self.para_bg = torch.tensor([1., 1., 1.], requires_grad=trainable_bg, device=self.device)

        self.pos_init_method = None

    def component_wise_path_init(self, pred, init_type: str = 'sparse'):
        assert self.target_img is not None  # gt

        if init_type == 'random':
            self.pos_init_method = RandomCoordInit(self.canvas_height, self.canvas_width)
        elif init_type == 'sparse':
            # when initialized for the first time, the render result is None
            if pred is None:
                pred = self.para_bg.view(1, -1, 1, 1).repeat(1, 1, self.canvas_height, self.canvas_width)
            # then pred is the render result
            self.pos_init_method = SparseCoordInit(pred, self.target_img)
        elif init_type == 'naive':
            if pred is None:
                pred = self.para_bg.view(1, -1, 1, 1).repeat(1, 1, self.canvas_height, self.canvas_width)
            self.pos_init_method = NaiveCoordInit(pred, self.target_img)
        else:
            raise NotImplementedError(f"'{init_type}' is not support.")

    def init_image(self, num_paths=0):
        self.cur_shapes, self.cur_shape_groups = [], []

        for i in range(num_paths):
            path, color_ref = self.get_path()
            self.shapes.append(path)
            self.cur_shapes.append(path)

            wref, href = color_ref
            wref = max(0, min(int(wref), self.canvas_width - 1))
            href = max(0, min(int(href), self.canvas_height - 1))
            fill_color_init = list(self.target_img[0, :, href, wref]) + [1.]
            fill_color_init = torch.FloatTensor(fill_color_init)
            stroke_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(self.shapes) - 1]),
                fill_color=None if self.train_stroke else fill_color_init,
                stroke_color=stroke_color_init if self.train_stroke else None
            )
            self.shape_groups.append(path_group)
            self.cur_shape_groups.append(path_group)

        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + self.para_bg * (1 - img[:, :, 3:4])
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_image(self, step: int = 0):
        img = self.render_warp(seed=step)
        img = img[:, :, 3:4] * img[:, :, :3] + self.para_bg * (1 - img[:, :, 3:4])
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_path(self):
        num_segments = self.num_segments
        num_control_points = [2] * num_segments

        points = []
        # init segment
        if self.segment_init == 'circle':
            radius = self.radius if self.radius is not None else np.random.uniform(0.5, 1)
            if self.pos_init_method is not None:
                center = self.pos_init_method()
            else:
                center = (random.random(), random.random())
            bias = center
            color_ref = copy.deepcopy(bias)

            avg_degree = 360 / (num_segments * 3)
            for i in range(0, num_segments * 3):
                point = (
                    np.cos(np.deg2rad(i * avg_degree)), np.sin(np.deg2rad(i * avg_degree))
                )
                points.append(point)

            points = torch.FloatTensor(points) * radius + torch.FloatTensor(bias).unsqueeze(dim=0)
        else:  # 'random' init
            p0 = self.pos_init_method()
            color_ref = copy.deepcopy(p0)
            points.append(p0)
            for j in range(num_segments):
                radius = self.radius
                p1 = (p0[0] + radius * np.random.uniform(-0.5, 0.5),
                      p0[1] + radius * np.random.uniform(-0.5, 0.5))
                p2 = (p1[0] + radius * np.random.uniform(-0.5, 0.5),
                      p1[1] + radius * np.random.uniform(-0.5, 0.5))
                p3 = (p2[0] + radius * np.random.uniform(-0.5, 0.5),
                      p2[1] + radius * np.random.uniform(-0.5, 0.5))
                points.append(p1)
                points.append(p2)
                if j < num_segments - 1:
                    points.append(p3)
                    p0 = p3
            points = torch.FloatTensor(points)

        path = pydiffvg.Path(
            num_control_points=torch.LongTensor(num_control_points),
            points=points,
            stroke_width=torch.tensor(float(self.stroke_width)) if self.train_stroke else torch.tensor(0.0),
            is_closed=True
        )

        self.strokes_counter += 1
        return path, color_ref

    def clip_curve_shape(self):
        for group in self.shape_groups:
            if self.train_stroke:
                group.stroke_color.data.clamp_(0.0, 1.0)
            else:
                group.fill_color.data.clamp_(0.0, 1.0)

    def calc_distance_weight(self, loss_weight_keep):
        shapes_forsdf = copy.deepcopy(self.cur_shapes)
        shape_groups_forsdf = copy.deepcopy(self.cur_shape_groups)
        for si in shapes_forsdf:
            si.stroke_width = torch.FloatTensor([0]).to(self.device)
        for sg_idx, sgi in enumerate(shape_groups_forsdf):
            sgi.fill_color = torch.FloatTensor([1, 1, 1, 1]).to(self.device)
            sgi.shape_ids = torch.LongTensor([sg_idx]).to(self.device)

        sargs_forsdf = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, shapes_forsdf, shape_groups_forsdf
        )
        _render = pydiffvg.RenderFunction.apply
        with torch.no_grad():
            im_forsdf = _render(self.canvas_width,  # width
                                self.canvas_height,  # height
                                2,  # num_samples_x
                                2,  # num_samples_y
                                0,  # seed
                                None,
                                *sargs_forsdf)

        # use alpha channel is a trick to get 0-1 image
        im_forsdf = (im_forsdf[:, :, 3]).detach().cpu().numpy()
        loss_weight = get_sdf(im_forsdf, normalize='to1')
        loss_weight += loss_weight_keep
        loss_weight = np.clip(loss_weight, 0, 1)
        loss_weight = torch.FloatTensor(loss_weight).to(self.device)
        return loss_weight

    def set_parameters(self):
        # stroke`s location optimization
        self.points_vars = []
        for i, path in enumerate(self.cur_shapes):
            path.points.requires_grad = True
            self.points_vars.append(path.points)

            if self.train_stroke:
                path.stroke_width.requires_grad = True
                self.stroke_width_vars.append(path.stroke_width)

        # for stroke' color optimization
        self.color_vars = []
        for i, group in enumerate(self.cur_shape_groups):
            if self.train_stroke:
                group.stroke_color.requires_grad = True
                self.stroke_color_vars.append(group.stroke_color)
            else:
                group.fill_color.requires_grad = True
                self.color_vars.append(group.fill_color)

    def get_point_parameters(self):
        return self.points_vars

    def get_color_parameters(self):
        return self.color_vars

    def get_stroke_parameters(self):
        return self.stroke_width_vars, self.stroke_color_vars

    def get_bg_parameters(self):
        return self.para_bg

    def save_svg(self, fpath):
        pydiffvg.save_svg(f'{fpath}',
                          self.canvas_width,
                          self.canvas_height,
                          self.shapes,
                          self.shape_groups)


def get_sdf(phi, **kwargs):
    import skfmm  # local import

    phi = (phi - 0.5) * 2
    if (phi.max() <= 0) or (phi.min() >= 0):
        return np.zeros(phi.shape).astype(np.float32)
    sd = skfmm.distance(phi, dx=1)

    flip_negative = kwargs.get('flip_negative', True)
    if flip_negative:
        sd = np.abs(sd)

    truncate = kwargs.get('truncate', 10)
    sd = np.clip(sd, -truncate, truncate)
    # print(f"max sd value is: {sd.max()}")

    zero2max = kwargs.get('zero2max', True)
    if zero2max and flip_negative:
        sd = sd.max() - sd
    elif zero2max:
        raise ValueError

    normalize = kwargs.get('normalize', 'sum')
    if normalize == 'sum':
        sd /= sd.sum()
    elif normalize == 'to1':
        sd /= sd.max()
    return sd


class SparseCoordInit:

    def __init__(self, pred, gt, format='[bs x c x 2D]', quantile_interval=200, nodiff_thres=0.1):
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()

        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0]) ** 2).sum(0)
            self.reference_gt = copy.deepcopy(np.transpose(gt[0], (1, 2, 0)))
        elif format == ['[2D x c]']:
            self.map = (np.abs(pred - gt)).sum(-1)
            self.reference_gt = copy.deepcopy(gt[0])
        else:
            raise ValueError

        # OptionA: Zero too small errors to avoid the error too small deadloop
        self.map[self.map < nodiff_thres] = 0
        quantile_interval = np.linspace(0., 1., quantile_interval)
        quantized_interval = np.quantile(self.map, quantile_interval)
        # remove redundant
        quantized_interval = np.unique(quantized_interval)
        quantized_interval = sorted(quantized_interval[1:-1])
        self.map = np.digitize(self.map, quantized_interval, right=False)
        self.map = np.clip(self.map, 0, 255).astype(np.uint8)
        self.idcnt = {}
        for idi in sorted(np.unique(self.map)):
            self.idcnt[idi] = (self.map == idi).sum()
        # remove smallest one to remove the correct region
        self.idcnt.pop(min(self.idcnt.keys()))

    def __call__(self):
        if len(self.idcnt) == 0:
            h, w = self.map.shape
            return [np.random.uniform(0, 1) * w, np.random.uniform(0, 1) * h]

        target_id = max(self.idcnt, key=self.idcnt.get)
        _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
            (self.map == target_id).astype(np.uint8),
            connectivity=4
        )
        # remove cid = 0, it is the invalid area
        csize = [ci[-1] for ci in cstats[1:]]
        target_cid = csize.index(max(csize)) + 1
        center = ccenter[target_cid][::-1]
        coord = np.stack(np.where(component == target_cid)).T
        dist = np.linalg.norm(coord - center, axis=1)
        target_coord_id = np.argmin(dist)
        coord_h, coord_w = coord[target_coord_id]

        # replace_sampling
        self.idcnt[target_id] -= max(csize)
        if self.idcnt[target_id] == 0:
            self.idcnt.pop(target_id)
        self.map[component == target_cid] = 0
        return [coord_w, coord_h]


class RandomCoordInit:
    def __init__(self, canvas_width, canvas_height):
        self.canvas_width, self.canvas_height = canvas_width, canvas_height

    def __call__(self):
        w, h = self.canvas_width, self.canvas_height
        return [np.random.uniform(0, 1) * w, np.random.uniform(0, 1) * h]


class NaiveCoordInit:
    def __init__(self, pred, gt, format='[bs x c x 2D]', replace_sampling=True):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0]) ** 2).sum(0)
        elif format == ['[2D x c]']:
            self.map = ((pred - gt) ** 2).sum(-1)
        else:
            raise ValueError
        self.replace_sampling = replace_sampling

    def __call__(self):
        coord = np.where(self.map == self.map.max())
        coord_h, coord_w = coord[0][0], coord[1][0]
        if self.replace_sampling:
            self.map[coord_h, coord_w] = -1
        return [coord_w, coord_h]


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


class PainterOptimizer:

    def __init__(self,
                 renderer: Painter,
                 num_iter: int,
                 lr_config: omegaconf.DictConfig,
                 trainable_stroke: bool = False,
                 trainable_bg: bool = False):
        self.renderer = renderer
        self.num_iter = num_iter
        self.trainable_stroke = trainable_stroke
        self.trainable_bg = trainable_bg

        self.lr_base = {
            'point': lr_config.point,
            'color': lr_config.color,
            'stroke_width': lr_config.stroke_width,
            'stroke_color': lr_config.stroke_color,
            'bg': lr_config.bg
        }

        self.learnable_params = []  # list[Dict]

        self.optimizer = None
        self.scheduler = None

    def init_optimizers(self):
        # optimizers
        params = {}
        self.renderer.set_parameters()
        params['point'] = self.renderer.get_point_parameters()
        if self.trainable_stroke:
            params['stroke_width'], params['stroke_color'] = self.renderer.get_stroke_parameters()
        else:
            params['color'] = self.renderer.get_color_parameters()

        if self.trainable_bg:
            params['bg'] = self.renderer.get_bg_parameters()

        self.learnable_params = [
            {'params': params[ki], 'lr': self.lr_base[ki]} for ki in sorted(params.keys())
        ]
        self.optimizer = torch.optim.Adam(self.learnable_params)
        # lr schedule
        lr_lambda_fn = LinearDecayLR(self.num_iter, 0.4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda_fn, last_epoch=-1)

    def update_params(self, name: str, value: torch.tensor):
        for param_group in self.learnable_params:
            if param_group.get('_id') == name:
                param_group['params'] = value

    def update_lr(self):
        self.scheduler.step()

    def zero_grad_(self):
        self.optimizer.zero_grad()

    def step_(self):
        self.optimizer.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
