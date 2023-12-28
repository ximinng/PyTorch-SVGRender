# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

from typing import Union, List, Tuple
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class CLIPScoreWrapper(nn.Module):

    def __init__(self,
                 clip_model_name: str,
                 download_root: str = None,
                 device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
                 jit: bool = False,
                 # additional params
                 visual_score: bool = False,
                 feats_loss_type: str = None,
                 feats_loss_weights: List[float] = None,
                 fc_loss_weight: float = None,
                 context_length: int = 77):
        super().__init__()

        import clip  # local import

        # check model info
        self.clip_model_name = clip_model_name
        self.device = device
        self.available_models = clip.available_models()
        assert clip_model_name in self.available_models, f"A model backbone: {clip_model_name} that does not exist"

        # load CLIP
        self.model, self.preprocess = clip.load(clip_model_name, device, jit=jit, download_root=download_root)
        self.model.eval()

        # load tokenize
        self.tokenize_fn = partial(clip.tokenize, context_length=context_length)

        # load CLIP visual
        self.visual_encoder = VisualEncoderWrapper(self.model, clip_model_name).to(device)
        self.visual_encoder.eval()

        # check loss weights
        self.visual_score = visual_score
        if visual_score:
            assert feats_loss_type in ["l1", "l2", "cosine"], f"{feats_loss_type} is not exist."
            if clip_model_name.startswith("ViT"): assert len(feats_loss_weights) == 12
            if clip_model_name.startswith("RN"): assert len(feats_loss_weights) == 5

            # load visual loss wrapper
            self.visual_loss_fn = CLIPVisualLossWrapper(self.visual_encoder, feats_loss_type,
                                                        feats_loss_weights,
                                                        fc_loss_weight)

    @property
    def input_resolution(self):
        return self.model.visual.input_resolution  # default: 224

    @property
    def resize(self):  # Resize only
        return transforms.Compose([self.preprocess.transforms[0]])

    @property
    def normalize(self):
        return transforms.Compose([
            self.preprocess.transforms[0],  # Resize
            self.preprocess.transforms[1],  # CenterCrop
            self.preprocess.transforms[-1],  # Normalize
        ])

    @property
    def norm_(self):  # Normalize only
        return transforms.Compose([self.preprocess.transforms[-1]])

    def encode_image_layer_wise(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        semantic_vec, feature_maps = self.visual_encoder(x)
        return semantic_vec, feature_maps

    def encode_text(self, text: Union[str, List[str]], norm: bool = True) -> torch.Tensor:
        tokens = self.tokenize_fn(text).to(self.device)
        text_features = self.model.encode_text(tokens)
        if norm:
            text_features = text_features.mean(axis=0, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features_norm
        return text_features

    def encode_image(self, image: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.model.encode_image(image)
        if norm:
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features_norm
        return image_features

    @torch.no_grad()
    def predict(self,
                image: torch.Tensor,
                text: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        image_features = self.model.encode_image(image)
        text_tokenize = self.tokenize_fn(text).to(self.device)
        text_features = self.model.encode_text(text_tokenize)
        logits_per_image, logits_per_text = self.model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return image_features, text_features, probs

    def compute_text_visual_distance(
            self, image: torch.Tensor, text: Union[str, List[str]]
    ) -> torch.Tensor:
        image_features = self.model.encode_image(image)
        text_tokenize = self.tokenize_fn(text).to(self.device)
        text_features = self.model.encode_text(text_tokenize)

        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        loss = - (image_features_norm @ text_features_norm.T)
        return loss.mean()

    def directional_loss(self, src_text, src_img, tar_text, tar_img, thresh=None):
        # delta img
        img_direction = (tar_img - src_img)
        img_direction_norm = img_direction / img_direction.norm(dim=-1, keepdim=True)
        # delta text
        text_direction = (1 * tar_text - src_text).repeat(tar_img.size(0), 1)
        text_direction_norm = text_direction / text_direction.norm(dim=-1, keepdim=True)
        # Directional CLIP Loss
        loss_dir = (1 - torch.cosine_similarity(img_direction_norm, text_direction_norm, dim=1))
        if thresh is not None:
            loss_dir[loss_dir < thresh] = 0  # set value=0 when lt 0
            return loss_dir.mean()
        else:
            return loss_dir.mean()

    def compute_visual_distance(
            self, x: torch.Tensor, y: torch.Tensor, clip_norm: bool = True,
    ) -> Tuple[torch.Tensor, List]:
        # return a fc loss and the list of feat loss
        assert self.visual_score is True
        assert x.shape == y.shape
        assert x.shape[-1] == self.input_resolution and x.shape[-2] == self.input_resolution
        assert y.shape[-1] == self.input_resolution and y.shape[-2] == self.input_resolution

        if clip_norm:
            return self.visual_loss_fn(self.normalize(x), self.normalize(y))
        else:
            return self.visual_loss_fn(x, y)


class VisualEncoderWrapper(nn.Module):
    """
    semantic features and layer by layer feature maps are obtained from CLIP visual encoder.
    """

    def __init__(self, clip_model: nn.Module, clip_model_name: str):
        super().__init__()
        self.clip_model = clip_model
        self.clip_model_name = clip_model_name

        if clip_model_name.startswith("ViT"):
            self.feature_maps = OrderedDict()
            for i in range(12):  # 12 ResBlocks in ViT visual transformer
                self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                    self.make_hook(i)
                )

        if clip_model_name.startswith("RN"):
            layers = list(self.clip_model.visual.children())
            init_layers = torch.nn.Sequential(*layers)[:8]
            self.layer1 = layers[8]
            self.layer2 = layers[9]
            self.layer3 = layers[10]
            self.layer4 = layers[11]
            self.att_pool2d = layers[12]

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                # LND -> NLD (B, 77, 768)
                self.feature_maps[name] = output.permute(1, 0, 2)
            else:
                self.feature_maps[name] = output

        return hook

    def _forward_vit(self, x: torch.Tensor) -> Tuple[torch.Tensor, List]:
        fc_feature = self.clip_model.encode_image(x).float()
        feature_maps = [self.feature_maps[k] for k in range(12)]

        # fc_feature len: 1 ,feature_maps len: 12
        return fc_feature, feature_maps

    def _forward_resnet(self, x: torch.Tensor) -> Tuple[torch.Tensor, List]:
        def stem(m, x):
            for conv, bn, relu in [(m.conv1, m.bn1, m.relu1), (m.conv2, m.bn2, m.relu2), (m.conv3, m.bn3, m.relu3)]:
                x = torch.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x

        x = x.type(self.clip_model.visual.conv1.weight.dtype)
        x = stem(self.clip_model.visual, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)

        # fc_features len: 1 ,feature_maps len: 5
        return y, [x, x1, x2, x3, x4]

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.clip_model_name.startswith("ViT"):
            fc_feat, visual_feat_maps = self._forward_vit(x)
        if self.clip_model_name.startswith("RN"):
            fc_feat, visual_feat_maps = self._forward_resnet(x)

        return fc_feat, visual_feat_maps


class CLIPVisualLossWrapper(nn.Module):
    """
    Visual Feature Loss + FC loss
    """

    def __init__(
            self,
            visual_encoder: nn.Module,
            feats_loss_type: str = None,
            feats_loss_weights: List[float] = None,
            fc_loss_weight: float = None,
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.feats_loss_weights = feats_loss_weights
        self.fc_loss_weight = fc_loss_weight

        self.layer_criterion = layer_wise_distance(feats_loss_type)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_fc_feature, x_feat_maps = self.visual_encoder(x)
        y_fc_feature, y_feat_maps = self.visual_encoder(y)

        # visual feature loss
        if sum(self.feats_loss_weights) == 0:
            feats_loss_list = [torch.tensor(0, device=x.device)]
        else:
            feats_loss = self.layer_criterion(x_feat_maps, y_feat_maps, self.visual_encoder.clip_model_name)
            feats_loss_list = []
            for layer, w in enumerate(self.feats_loss_weights):
                if w:
                    feats_loss_list.append(feats_loss[layer] * w)

        # visual fc loss, default: cosine similarity
        if self.fc_loss_weight == 0:
            fc_loss = torch.tensor(0, device=x.device)
        else:
            fc_loss = (1 - torch.cosine_similarity(x_fc_feature, y_fc_feature, dim=1)).mean()
            fc_loss = fc_loss * self.fc_loss_weight

        return fc_loss, feats_loss_list


#################################################################################
#                                layer wise metric                              #
#################################################################################

def layer_wise_distance(metric_name: str):
    return {
        "l1": l1_layer_wise,
        "l2": l2_layer_wise,
        "cosine": cosine_layer_wise
    }.get(metric_name.lower())


def l2_layer_wise(x_features, y_features, clip_model_name):
    return [
        torch.square(x_conv - y_conv).mean()
        for x_conv, y_conv in zip(x_features, y_features)
    ]


def l1_layer_wise(x_features, y_features, clip_model_name):
    return [
        torch.abs(x_conv - y_conv).mean()
        for x_conv, y_conv in zip(x_features, y_features)
    ]


def cosine_layer_wise(x_features, y_features, clip_model_name):
    if clip_model_name.startswith("RN"):
        return [
            (1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean()
            for x_conv, y_conv in zip(x_features, y_features)
        ]
    return [
        (1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean()
        for x_conv, y_conv in zip(x_features, y_features)
    ]
