from typing import Dict

import torch
import torch.nn as nn
from torch.nn import functional as nnf
import torchvision
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class ToneLoss(nn.Module):
    def __init__(self, cfg):
        super(ToneLoss, self).__init__()
        self.dist_loss_weight = cfg.dist_loss_weight
        self.im_init = None
        self.mse_loss = nn.MSELoss()
        self.blur = torchvision.transforms.GaussianBlur(
            kernel_size=(cfg.pixel_dist_kernel_blur,
                         cfg.pixel_dist_kernel_blur),
            sigma=(cfg.pixel_dist_sigma, cfg.pixel_dist_sigma)
        )
        self.init_blurred = None

    def set_image_init(self, im_init):
        self.im_init = im_init
        self.init_blurred = self.blur(self.im_init)

    def get_scheduler(self, step=None):
        if step is not None:
            return self.dist_loss_weight * np.exp(-(1 / 5) * ((step - 300) / (20)) ** 2)
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        blurred_cur = self.blur(cur_raster)
        return self.mse_loss(self.init_blurred.detach(), blurred_cur) * self.get_scheduler(step)


class ConformalLoss:
    def __init__(self, parameters, shape_groups, target_letter: str, device: torch.device):
        self.parameters = parameters
        self.device = device
        self.target_letter = target_letter
        self.shape_groups = shape_groups
        self.faces = self.init_faces(device)
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))]

        with torch.no_grad():
            self.angles = []
            self.reset(device)

    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_

    def get_letter_inds(self, letter_to_insert):
        for group, l in zip(self.shape_groups, self.target_letter):
            if l == letter_to_insert:
                letter_inds = group.shape_ids
                return letter_inds[0], letter_inds[-1], len(letter_inds)

    def reset(self, device):
        points = torch.cat([point.to(device) for point in self.parameters])
        self.angles = self.get_angles(points)

    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [
                self.parameters[i].clone().detach().cpu().numpy()
                for i in range(len(self.parameters))
            ]
            start_ind, end_ind, shapes_per_letter = self.get_letter_inds(c)
            print(c, "start_ind: ", start_ind.item(), ", end_ind: ", end_ind.item())
            holes = []
            if shapes_per_letter > 1:
                holes = points_np[start_ind + 1:end_ind]
            poly = Polygon(points_np[start_ind], holes=holes)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=bool)
            faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_

    def __call__(self) -> torch.Tensor:
        loss_angles = 0
        points = torch.cat(self.parameters).to(self.device)
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            loss_angles += (nnf.mse_loss(angles[i], self.angles[i]))
        return loss_angles
