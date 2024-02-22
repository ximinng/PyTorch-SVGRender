# -*- coding: utf-8 -*-
# Author: ximing
# Description: DiffVG painter and optimizer
# Copyright (c) 2023, XiMing Xing.
# License: MPL-2.0 License

import copy
import random
from typing import List

import omegaconf
import numpy as np
import pydiffvg
import torch
from torch.optim.lr_scheduler import LambdaLR

from pytorch_svgrender.diffvg_warp import DiffVGState


class Painter(DiffVGState):

    def __init__(
            self,
            target_img: torch.Tensor,
            diffvg_cfg: omegaconf.DictConfig,
            canvas_size: List,
            path_type: str = 'unclosed',
            max_width: float = 3.0,
            device: torch.device = None,
    ):
        super(Painter, self).__init__(device, print_timing=diffvg_cfg.print_timing,
                                      canvas_width=canvas_size[0], canvas_height=canvas_size[1])

        self.target_img = target_img
        self.path_type: str = path_type
        self.max_width = max_width
        self.train_stroke: bool = path_type == 'unclosed'

        self.strokes_counter: int = 0  # counts the number of calls to "get_path"

    def init_image(self, num_paths=0):
        for i in range(num_paths):
            path = self.get_path()
            self.shapes.append(path)
            self.shapes.append(path)

            fill_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
            stroke_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(self.shapes) - 1]),
                fill_color=None if self.train_stroke else fill_color_init,
                stroke_color=stroke_color_init if self.train_stroke else None
            )
            self.shape_groups.append(path_group)
            self.shape_groups.append(path_group)

        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) \
              * (1 - img[:, :, 3:4])
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_image(self, step: int = 0):
        img = self.render_warp(seed=step)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) \
              * (1 - img[:, :, 3:4])
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_path(self):
        if self.path_type == 'unclosed':
            num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= self.canvas_width
            points[:, 1] *= self.canvas_height
            # points = torch.rand(3 * num_segments + 1, 2) * min(canvas_width, canvas_height)

            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 stroke_width=torch.tensor(1.0),
                                 is_closed=False)
        elif self.path_type == 'closed':
            num_segments = random.randint(3, 5)
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                if j < num_segments - 1:
                    points.append(p3)
                    p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= self.canvas_width
            points[:, 1] *= self.canvas_height
            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 stroke_width=torch.tensor(1.0),
                                 is_closed=True)

        self.strokes_counter += 1
        return path

    def clip_curve_shape(self):
        if self.train_stroke:  # open-form path
            for path in self.shapes:
                path.stroke_width.data.clamp_(1.0, self.max_width)
            for group in self.shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)
        else:  # closed-form path
            for group in self.shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)

    def set_parameters(self):
        # stroke`s location optimization
        self.point_vars = []
        for i, path in enumerate(self.shapes):
            path.points.requires_grad = True
            self.point_vars.append(path.points)

            if self.train_stroke:
                path.stroke_width.requires_grad = True
                self.width_vars.append(path.stroke_width)

        # for stroke' color optimization
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if self.train_stroke:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)
            else:
                group.fill_color.requires_grad = True
                self.color_vars.append(group.fill_color)

    def get_point_parameters(self):
        return self.point_vars

    def get_color_parameters(self):
        return self.color_vars

    def get_stroke_parameters(self):
        return self.width_vars, self.get_color_parameters()

    def save_svg(self, fpath):
        pydiffvg.save_svg(f'{fpath}', self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)


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
                 trainable_stroke: bool = False):
        self.renderer = renderer
        self.num_iter = num_iter
        self.trainable_stroke = trainable_stroke

        self.lr_base = {
            'point': lr_config.point,
            'color': lr_config.color,
            'stroke_width': lr_config.stroke_width,
            'stroke_color': lr_config.stroke_color,
        }

        self.learnable_params = []  # list[Dict]

        self.optimizer = None
        self.scheduler = None

    def init_optimizer(self):
        # optimizers
        params = {}
        self.renderer.set_parameters()
        params['point'] = self.renderer.get_point_parameters()

        if self.trainable_stroke:
            params['stroke_width'], params['stroke_color'] = self.renderer.get_stroke_parameters()
        else:
            params['color'] = self.renderer.get_color_parameters()

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
