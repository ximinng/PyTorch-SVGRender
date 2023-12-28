# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import random
import pathlib

import omegaconf
import pydiffvg
import numpy as np
import torch

from pytorch_svgrender.libs.modules.edge_map.DoG import XDoG
from pytorch_svgrender.diffvg_warp import DiffVGState


class Painter(DiffVGState):

    def __init__(
            self,
            cfg: omegaconf.DictConfig,
            diffvg_cfg: omegaconf.DictConfig,
            num_strokes: int = 4,
            num_segments: int = 4,
            canvas_size: int = 224,
            device: torch.device = None,
            target_im: torch.Tensor = None,
            attention_map: torch.Tensor = None,
            mask: torch.Tensor = None,
    ):
        super(Painter, self).__init__(device, print_timing=diffvg_cfg.print_timing,
                                      canvas_width=canvas_size, canvas_height=canvas_size)

        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = cfg.width
        self.max_width = cfg.max_width
        self.optim_width = cfg.optim_width
        self.control_points_per_seg = cfg.control_points_per_seg
        self.optim_rgba = cfg.optim_rgba
        self.optim_alpha = cfg.optim_opacity
        self.num_stages = cfg.num_stages
        self.softmax_temp = cfg.softmax_temp

        self.shapes = []
        self.shape_groups = []
        self.num_control_points = 0
        self.color_vars_threshold = cfg.color_vars_threshold

        self.path_svg = cfg.path_svg
        self.strokes_per_stage = self.num_paths
        self.optimize_flag = []

        # attention related for strokes initialisation
        self.attention_init = cfg.attention_init
        self.xdog_intersec = cfg.xdog_intersec

        self.GT_input = target_im
        self.mask = mask
        self.attention_map = attention_map if self.attention_init else None

        self.thresh = self.set_attention_threshold_map() if self.attention_init else None
        self.strokes_counter = 0  # counts the number of calls to "get_path"

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
        img = img[:, :, 3:4] * img[:, :, :3] + \
              torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW

        return img

    def get_image(self):
        img = self.render_warp()

        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_path(self):
        self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        points = []
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

    def clip_curve_shape(self):
        if self.optim_width:
            for path in self.shapes:
                path.stroke_width.data.clamp_(1.0, self.max_width)
        if self.optim_rgba:
            for group in self.shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)
        else:
            if self.optim_alpha:
                for group in self.shape_groups:
                    # group.stroke_color.data: RGBA
                    group.stroke_color.data[:3].clamp_(0., 0.)  # to force black stroke
                    group.stroke_color.data[-1].clamp_(0., 1.)  # opacity

    def path_pruning(self):
        for group in self.shape_groups:
            group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()

    def set_points_parameters(self):
        # stoke`s location optimization
        self.point_vars = []
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.requires_grad = True
                self.point_vars.append(path.points)

    def get_points_params(self):
        return self.point_vars

    def set_width_parameters(self):
        # stroke`s  width optimization
        self.width_vars = []
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.stroke_width.requires_grad = True
                self.width_vars.append(path.stroke_width)

    def get_width_parameters(self):
        return self.width_vars

    def set_color_parameters(self):
        # for strokes color optimization (opacity)
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if self.optimize_flag[i]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)

    def get_color_parameters(self):
        return self.color_vars

    def save_svg(self, output_dir, fname):
        pydiffvg.save_svg(f'{output_dir}/{fname}.svg',
                          self.canvas_width,
                          self.canvas_height,
                          self.shapes,
                          self.shape_groups)

    @staticmethod
    def softmax(x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()

    def set_inds_ldm(self):
        attn_map = (self.attention_map - self.attention_map.min()) / \
                   (self.attention_map.max() - self.attention_map.min())

        if self.xdog_intersec:
            xdog = XDoG(k=10)
            im_xdog = xdog(self.GT_input[0].permute(1, 2, 0).cpu().numpy())
            print(f"use XDoG, shape: {im_xdog.shape}")
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map

        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)

        # select points
        k = self.num_stages * self.num_paths
        self.inds = np.random.choice(range(attn_map.flatten().shape[0]),
                                     size=k,
                                     replace=False,
                                     p=attn_map_soft.flatten())
        self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return attn_map_soft

    def set_attention_threshold_map(self):
        return self.set_inds_ldm()

    def get_attn(self):
        return self.attention_map

    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds

    def get_mask(self):
        return self.mask


class SketchPainterOptimizer:

    def __init__(
            self,
            renderer: Painter,
            points_lr: float,
            optim_alpha: bool,
            optim_rgba: bool,
            color_lr: float,
            optim_width: bool,
            width_lr: float
    ):
        self.renderer = renderer

        self.points_lr = points_lr
        self.optim_color = optim_alpha or optim_rgba
        self.color_lr = color_lr
        self.optim_width = optim_width
        self.width_lr = width_lr

        self.points_optimizer, self.width_optimizer, self.color_optimizer = None, None, None

    def init_optimizers(self):
        self.renderer.set_points_parameters()
        self.points_optimizer = torch.optim.Adam(self.renderer.get_points_params(), lr=self.points_lr)
        if self.optim_color:
            self.renderer.set_color_parameters()
            self.color_optimizer = torch.optim.Adam(self.renderer.get_color_parameters(), lr=self.color_lr)
        if self.optim_width:
            self.renderer.set_width_parameters()
            self.width_optimizer = torch.optim.Adam(self.renderer.get_width_parameters(), lr=self.width_lr)

    def update_lr(self, step, decay_steps=(500, 750)):
        if step % decay_steps[0] == 0 and step > 0:
            for param_group in self.points_optimizer.param_groups:
                param_group['lr'] = 0.4
        if step % decay_steps[1] == 0 and step > 0:
            for param_group in self.points_optimizer.param_groups:
                param_group['lr'] = 0.1

    def zero_grad_(self):
        self.points_optimizer.zero_grad()
        if self.optim_color:
            self.color_optimizer.zero_grad()
        if self.optim_width:
            self.width_optimizer.zero_grad()

    def step_(self):
        self.points_optimizer.step()
        if self.optim_color:
            self.color_optimizer.step()
        if self.optim_width:
            self.width_optimizer.step()

    def get_lr(self):
        return self.points_optimizer.param_groups[0]['lr']
