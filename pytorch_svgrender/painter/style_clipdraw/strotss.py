# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import math

import torch
import torch.nn as nn
import torchvision
import numpy as np


class VGG16Extractor(nn.Module):
    def __init__(self, space):
        super().__init__()
        # load pretrained model
        self.vgg_layers = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.DEFAULT
        ).features

        for param in self.parameters():
            param.requires_grad = False
        self.capture_layers = [1, 3, 6, 8, 11, 13, 15, 22, 29]
        self.space = space

    def forward_base(self, x):
        feat = [x]
        for i in range(len(self.vgg_layers)):
            x = self.vgg_layers[i](x)
            if i in self.capture_layers:
                feat.append(x)
        return feat

    def forward(self, x):
        if self.space != 'vgg':
            x = (x + 1.) / 2.
            x = x - (torch.Tensor([0.485, 0.456, 0.406]).to(x.device).view(1, -1, 1, 1))
            x = x / (torch.Tensor([0.229, 0.224, 0.225]).to(x.device).view(1, -1, 1, 1))
        feat = self.forward_base(x)
        return feat

    def forward_samples_hypercolumn(self, X, samps=100):
        feat = self.forward(X)

        xx, xy = np.meshgrid(np.arange(X.shape[2]), np.arange(X.shape[3]))
        xx = np.expand_dims(xx.flatten(), 1)
        xy = np.expand_dims(xy.flatten(), 1)
        xc = np.concatenate([xx, xy], 1)

        samples = min(samps, xc.shape[0])

        np.random.shuffle(xc)
        xx = xc[:samples, 0]
        yy = xc[:samples, 1]

        feat_samples = []
        for i in range(len(feat)):

            layer_feat = feat[i]

            # hack to detect lower resolution
            if i > 0 and feat[i].size(2) < feat[i - 1].size(2):
                xx = xx / 2.0
                yy = yy / 2.0

            xx = np.clip(xx, 0, layer_feat.shape[2] - 1).astype(np.int32)
            yy = np.clip(yy, 0, layer_feat.shape[3] - 1).astype(np.int32)

            features = layer_feat[:, :, xx[range(samples)], yy[range(samples)]]
            feat_samples.append(features.clone().detach())

        feat = torch.cat(feat_samples, 1)
        return feat


class StyleLoss:

    def spatial_feature_extract(self, feat_result, feat_content, xx, xy):
        l2, l3 = [], []
        device = feat_result[0].device

        # for each extracted layer
        for i in range(len(feat_result)):
            fr = feat_result[i]
            fc = feat_content[i]

            # hack to detect reduced scale
            if i > 0 and feat_result[i - 1].size(2) > feat_result[i].size(2):
                xx = xx / 2.0
                xy = xy / 2.0

            # go back to ints and get residual
            xxm = np.floor(xx).astype(np.float32)
            xxr = xx - xxm

            xym = np.floor(xy).astype(np.float32)
            xyr = xy - xym

            # do bilinear resample
            w00 = torch.from_numpy((1. - xxr) * (1. - xyr)).float().view(1, 1, -1, 1).to(device)
            w01 = torch.from_numpy((1. - xxr) * xyr).float().view(1, 1, -1, 1).to(device)
            w10 = torch.from_numpy(xxr * (1. - xyr)).float().view(1, 1, -1, 1).to(device)
            w11 = torch.from_numpy(xxr * xyr).float().view(1, 1, -1, 1).to(device)

            xxm = np.clip(xxm.astype(np.int32), 0, fr.size(2) - 1)
            xym = np.clip(xym.astype(np.int32), 0, fr.size(3) - 1)

            s00 = xxm * fr.size(3) + xym
            s01 = xxm * fr.size(3) + np.clip(xym + 1, 0, fr.size(3) - 1)
            s10 = np.clip(xxm + 1, 0, fr.size(2) - 1) * fr.size(3) + (xym)
            s11 = np.clip(xxm + 1, 0, fr.size(2) - 1) * fr.size(3) + np.clip(xym + 1, 0, fr.size(3) - 1)

            fr = fr.view(1, fr.size(1), fr.size(2) * fr.size(3), 1)
            fr = fr[:, :, s00, :].mul_(w00).add_(fr[:, :, s01, :].mul_(w01)).add_(fr[:, :, s10, :].mul_(w10)).add_(
                fr[:, :, s11, :].mul_(w11))

            fc = fc.view(1, fc.size(1), fc.size(2) * fc.size(3), 1)
            fc = fc[:, :, s00, :].mul_(w00).add_(fc[:, :, s01, :].mul_(w01)).add_(fc[:, :, s10, :].mul_(w10)).add_(
                fc[:, :, s11, :].mul_(w11))

            l2.append(fr)
            l3.append(fc)

        x_st = torch.cat([li.contiguous() for li in l2], 1)
        c_st = torch.cat([li.contiguous() for li in l3], 1)

        xx = torch.from_numpy(xx).view(1, 1, x_st.size(2), 1).float().to(device)
        yy = torch.from_numpy(xy).view(1, 1, x_st.size(2), 1).float().to(device)

        x_st = torch.cat([x_st, xx, yy], 1)
        c_st = torch.cat([c_st, xx, yy], 1)
        return x_st, c_st

    def rgb_to_yuv(self, rgb):
        C = torch.Tensor(
            [[0.577350, 0.577350, 0.577350], [-0.577350, 0.788675, -0.211325], [-0.577350, -0.211325, 0.788675]]
        ).to(rgb.device)
        yuv = torch.mm(C, rgb)
        return yuv

    def pairwise_distances_cos(self, x, y):
        x_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1))
        y_t = torch.transpose(y, 0, 1)
        y_norm = torch.sqrt((y ** 2).sum(1).view(1, -1))
        dist = 1. - torch.mm(x, y_t) / x_norm / y_norm
        return dist

    def pairwise_distances_sq_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 1e-5, 1e5) / x.size(1)

    def distmat(self, x, y, cos_d=True):
        if cos_d:
            M = self.pairwise_distances_cos(x, y)
        else:
            M = torch.sqrt(self.pairwise_distances_sq_l2(x, y))
        return M

    def style_loss(self, X, Y):
        d = X.shape[1]

        if d == 3:
            X = self.rgb_to_yuv(X.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
            Y = self.rgb_to_yuv(Y.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        else:
            X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
            Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

        # Relaxed EMD
        CX_M = self.distmat(X, Y, cos_d=True)

        if d == 3:
            CX_M = CX_M + self.distmat(X, Y, cos_d=False)

        m1, m1_inds = CX_M.min(1)
        m2, m2_inds = CX_M.min(0)

        remd = torch.max(m1.mean(), m2.mean())

        return remd

    def moment_loss(self, X, Y, moments=[1, 2]):
        loss = 0.
        X = X.squeeze().t()
        Y = Y.squeeze().t()

        mu_x = torch.mean(X, 0, keepdim=True)
        mu_y = torch.mean(Y, 0, keepdim=True)
        mu_d = torch.abs(mu_x - mu_y).mean()

        if 1 in moments:
            loss = loss + mu_d

        if 2 in moments:
            X_c = X - mu_x
            Y_c = Y - mu_y
            X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0] - 1)
            Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0] - 1)

            D_cov = torch.abs(X_cov - Y_cov).mean()
            loss = loss + D_cov

        return loss

    def forward(self, feat_result, feat_content, feat_style, indices, content_weight, moment_weight=1.0):
        # spatial feature extract
        num_locations = 1024
        spatial_result, spatial_content = self.spatial_feature_extract(
            feat_result, feat_content, indices[0][:num_locations], indices[1][:num_locations]
        )

        # loss_content = content_loss(spatial_result, spatial_content)

        d = feat_style.shape[1]
        spatial_style = feat_style.view(1, d, -1, 1)
        feat_max = 3 + 2 * 64 + 128 * 2 + 256 * 3 + 512 * 2  # (sum of all extracted channels)

        loss_remd = self.style_loss(spatial_result[:, :feat_max, :, :], spatial_style[:, :feat_max, :, :])

        loss_moment = self.moment_loss(spatial_result[:, :-2, :, :],
                                       spatial_style,
                                       moments=[1, 2])  # -2 is so that it can fit?
        # palette matching
        content_weight_frac = 1. / max(content_weight, 1.)
        loss_moment += content_weight_frac * self.style_loss(spatial_result[:, :3, :, :], spatial_style[:, :3, :, :])

        loss_style = loss_remd + moment_weight * loss_moment
        # print(f'Style: {loss_style.item():.3f}, Content: {loss_content.item():.3f}')

        style_weight = 1.0 + moment_weight
        loss_total = (loss_style) / (content_weight + style_weight)
        return loss_total


def sample_indices(feat_content, feat_style):
    const = 128 ** 2  # 32k or so
    big_size = feat_content.shape[2] * feat_content.shape[3]  # num feaxels

    stride_x = int(max(math.floor(math.sqrt(big_size // const)), 1))
    offset_x = np.random.randint(stride_x)
    stride_y = int(max(math.ceil(math.sqrt(big_size // const)), 1))
    offset_y = np.random.randint(stride_y)
    xx, xy = np.meshgrid(
        np.arange(feat_content.shape[2])[offset_x::stride_x],
        np.arange(feat_content.shape[3])[offset_y::stride_y]
    )
    xx = xx.flatten()
    xy = xy.flatten()
    return xx, xy
