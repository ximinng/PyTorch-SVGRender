# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description: SVG Painter and ist optimizer

from typing import Tuple

import omegaconf
import pydiffvg
import torch
import numpy as np

from pytorch_svgrender.diffvg_warp import DiffVGState
from pytorch_svgrender.utils import get_rgb_from_color


class Painter(DiffVGState):

    def __init__(self, device=None):
        super().__init__(device)
        self.device = device

        self.strokes_counter = 0  # num of paths

    def init_shapes(self, path_svg, reinit_cfg: omegaconf.DictConfig = None):
        print(f"-> init svg from `{path_svg}` ...")
        self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = self.load_svg(path_svg)
        self.strokes_counter = len(self.shapes)

        """re-init font color"""
        if reinit_cfg is not None:
            self.color_init(reinit_cfg)

        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + \
              torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def color_init(self, reinit_cfg: omegaconf.DictConfig):
        if not reinit_cfg.reinit:
            return

        if reinit_cfg.reinit_color == 'randn':
            for i, group in enumerate(self.shape_groups):
                color_val = np.random.random(size=3).tolist() + [1.0]
                group.fill_color = torch.FloatTensor(color_val)
        elif reinit_cfg.reinit_color == 'randn_all':
            color_val = np.random.random(size=3).tolist() + [1.0]
            for i, group in enumerate(self.shape_groups):
                group.fill_color = torch.FloatTensor(color_val)
        else:
            rgb = get_rgb_from_color(str(reinit_cfg.reinit_color))
            color_val = list(rgb) + [1.0]
            for i, group in enumerate(self.shape_groups):
                group.fill_color = torch.FloatTensor(color_val)

    def clip_curve_shape(self):
        for group in self.shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)
            # force opacity
            group.fill_color.data[-1] = 1.0

    def get_image(self):
        img = self.render_warp()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def set_parameters(self):
        self.point_vars = []
        # the strokes point optimization
        for i, path in enumerate(self.shapes):
            path.points.requires_grad = True
            self.point_vars.append(path.points)

        # the strokes color optimization
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if group.fill_color is not None:
                group.fill_color.requires_grad = True
                self.color_vars.append(group.fill_color)
            if group.stroke_color is not None:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)

    def get_point_parameters(self):
        return self.point_vars

    def get_color_parameters(self):
        return self.color_vars

    def pretty_save_svg(self, filename, width=None, height=None, shapes=None, shape_groups=None):
        width = self.canvas_width if width is None else width
        height = self.canvas_height if height is None else height
        shapes = self.shapes if shapes is None else shapes
        shape_groups = self.shape_groups if shape_groups is None else shape_groups

        self.save_svg(filename, width, height, shapes, shape_groups, use_gamma=False, background=None)

    def load_svg(self, path_svg):
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(path_svg)
        return canvas_width, canvas_height, shapes, shape_groups


class PainterOptimizer:

    def __init__(self, renderer: Painter, lr_cfg: omegaconf.DictConfig):
        self.renderer = renderer
        self.point_lr = lr_cfg.point
        self.color_lr = lr_cfg.color
        self.point_optimizer, self.color_optimizer = None, None

    def init_optimizers(self):
        self.renderer.set_parameters()
        self.point_optimizer = torch.optim.Adam([
            {'params': self.renderer.get_point_parameters(), 'lr': self.point_lr}])
        self.color_optimizer = torch.optim.Adam([
            {'params': self.renderer.get_color_parameters(), 'lr': self.color_lr}])

    def update_lr(self, step):
        pass

    def zero_grad_(self):
        self.point_optimizer.zero_grad()
        self.color_optimizer.zero_grad()

    def step_(self):
        self.point_optimizer.step()
        self.color_optimizer.step()

    def get_lr(self) -> Tuple[float, float]:
        return self.point_optimizer.param_groups[0]['lr'], self.color_optimizer.param_groups[0]['lr']
