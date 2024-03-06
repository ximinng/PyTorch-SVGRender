import random
import pathlib

import omegaconf
import pydiffvg
import torch

from pytorch_svgrender.diffvg_warp import DiffVGState


class Painter(DiffVGState):

    def __init__(
            self,
            method_cfg: omegaconf.DictConfig,
            diffvg_cfg: omegaconf.DictConfig,
            num_strokes: int = 4,
            canvas_size: int = 224,
            device: torch.device = None,
    ):
        super(Painter, self).__init__(device, print_timing=diffvg_cfg.print_timing,
                                      canvas_width=canvas_size, canvas_height=canvas_size)
        self.method_cfg = method_cfg

        self.num_paths = num_strokes
        self.max_width = method_cfg.max_width
        self.num_stages = method_cfg.num_stages

        self.black_stroke_color = method_cfg.black_stroke_color

        self.path_svg = method_cfg.path_svg
        self.strokes_per_stage = self.num_paths
        self.optimize_flag = []

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
                print(f"-> init svg from '{self.path_svg}' ...")

                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = self.load_svg(self.path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_paths_exists = len(self.shapes)

            for i in range(num_paths_exists, self.num_paths):
                if self.black_stroke_color:
                    stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                else:
                    stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()])
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

    def get_image(self, step=0):
        img = self.render_warp(step)
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_path(self):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)

        for j in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height

        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points,
                             stroke_width=torch.tensor(1.0),
                             is_closed=False)
        self.strokes_counter += 1
        return path

    def clip_curve_shape(self):
        for path in self.shapes:
            path.stroke_width.data.clamp_(1.0, self.max_width)
        for group in self.shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)

    def set_parameters(self):
        # stroke`s location and width optimization
        self.point_vars = []
        self.width_vars = []
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.requires_grad = True
                self.point_vars.append(path.points)
                path.stroke_width.requires_grad = True
                self.width_vars.append(path.stroke_width)

        # for stroke' color optimization
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if self.optimize_flag[i]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)

        return self.point_vars, self.width_vars, self.color_vars

    def learnable_parameters(self):
        return self.point_vars + self.width_vars + self.color_vars

    def save_svg(self, output_dir, name):
        pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name),
                          self.canvas_width, self.canvas_height,
                          self.shapes, self.shape_groups)


class PainterOptimizer:

    def __init__(self, renderer: Painter, points_lr: float, width_lr: float, color_lr: float):
        self.renderer = renderer

        self.points_lr = points_lr
        self.width_lr = width_lr
        self.color_lr = color_lr

        self.points_optimizer, self.width_optimizer, self.color_optimizer = None, None, None

    def init_optimizers(self):
        point_vars, width_vars, color_vars = self.renderer.set_parameters()
        self.points_optimizer = torch.optim.Adam(point_vars, lr=self.points_lr)
        self.width_optimizer = torch.optim.Adam(width_vars, lr=self.width_lr)
        self.color_optimizer = torch.optim.Adam(color_vars, lr=self.color_lr)

    def update_lr(self, step, decay_steps=(500, 750)):
        if step % decay_steps[0] == 0:
            for param_group in self.points_optimizer.param_groups:
                param_group['lr'] = 0.4
        if step % decay_steps[1] == 0:
            for param_group in self.points_optimizer.param_groups:
                param_group['lr'] = 0.1

    def zero_grad_(self):
        self.points_optimizer.zero_grad()
        self.width_optimizer.zero_grad()
        self.color_optimizer.zero_grad()

    def step_(self):
        self.points_optimizer.step()
        self.width_optimizer.step()
        self.color_optimizer.step()

    def get_lr(self):
        return self.points_optimizer.param_groups[0]['lr']
