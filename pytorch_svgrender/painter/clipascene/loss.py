import collections
import re

import clip
import torch
import torch.nn as nn
from torchvision import models, transforms


def compute_grad_norm_losses(losses_dict, model, points_mlp):
    '''
    Balances multiple losses by weighting them inversly proportional
    to their overall gradient contribution.
    
    Args:
        losses: A dictionary of losses.
        model: A PyTorch model.
    Returns:
        A dictionary of loss weights.
    '''
    grad_norms = {}
    for loss_name, loss in losses_dict.items():
        loss.backward(retain_graph=True)
        grad_sum = sum([w.grad.abs().sum().item() for w in model.parameters() if w.grad is not None])
        num_elem = sum([w.numel() for w in model.parameters() if w.grad is not None])
        grad_norms[loss_name] = grad_sum / num_elem
        model.zero_grad()
        points_mlp.zero_grad()

    grad_norms_total = sum(grad_norms.values())

    loss_weights = {}
    for loss_name, loss in losses_dict.items():
        weight = (grad_norms_total - grad_norms[loss_name]) / ((len(losses_dict) - 1) * grad_norms_total)
        loss_weights[loss_name] = weight

    return loss_weights


class Loss(nn.Module):
    def __init__(self, args, mask=None, device="cpu"):
        super(Loss, self).__init__()
        self.args = args
        self.percep_loss = args.percep_loss
        self.device = device

        self.train_with_clip = args.train_with_clip
        self.clip_weight = args.clip_weight
        self.start_clip = args.start_clip

        self.clip_conv_loss = args.clip_conv_loss
        self.clip_mask_loss = args.clip_mask_loss
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.clip_text_guide = args.clip_text_guide
        self.width_optim = args.width_optim
        self.width_loss_weight = args.width_loss_weight
        self.ratio_loss = args.ratio_loss
        if isinstance(args.clip_conv_layer_weights, str):
            self.args.clip_conv_layer_weights = [
                float(item) for item in args.clip_conv_layer_weights.split(',')
            ]

        self.losses_to_apply = self.get_losses_to_apply()
        self.gradnorm = args.gradnorm
        if args.gradnorm:
            self.new_weights = {}

        self.loss_mapper = {}
        if self.clip_conv_loss:
            self.loss_mapper["clip_conv_loss"] = CLIPConvLoss(args, mask, device)
        if self.clip_mask_loss:
            self.loss_mapper["clip_mask_loss"] = CLIPmaskLoss(args, mask, device)
        if self.width_optim:
            self.loss_mapper["width_loss"] = WidthLoss(args, device)
        if self.ratio_loss:
            self.loss_mapper["ratio_loss"] = RatioLoss(args, device)

    def get_losses_to_apply(self):
        losses_to_apply = []
        if self.percep_loss != "none":
            losses_to_apply.append(self.percep_loss)
        if self.train_with_clip and self.start_clip == 0:
            losses_to_apply.append("clip")
        if self.clip_conv_loss:
            losses_to_apply.append("clip_conv_loss")
        if self.clip_mask_loss:
            losses_to_apply.append("clip_mask_loss")
        if self.clip_text_guide:
            losses_to_apply.append("clip_text")
        if self.width_optim:
            losses_to_apply.append("width_loss")
        if self.ratio_loss:
            losses_to_apply.append("ratio_loss")
        return losses_to_apply

    def update_losses_to_apply(self, epoch, width_opt=None, mode="train"):
        if "clip" not in self.losses_to_apply:
            if self.train_with_clip:
                if epoch > self.start_clip:
                    self.losses_to_apply.append("clip")
        # for width loss switch
        if width_opt is not None:
            if self.width_optim and "width_loss" not in self.losses_to_apply and mode == "eval":
                self.losses_to_apply.append("width_loss")
            if width_opt and "width_loss" not in self.losses_to_apply:
                self.losses_to_apply.append("width_loss")
            if not width_opt and "width_loss" in self.losses_to_apply and mode == "train":
                self.losses_to_apply.remove("width_loss")

    def forward(self, sketches, targets, epoch, widths=None, renderer=None, optimizer=None, mode="train",
                width_opt=None):
        loss = 0
        self.update_losses_to_apply(epoch, width_opt, mode)

        losses_dict = {}
        loss_coeffs = {}
        if self.width_optim:
            loss_coeffs["width_loss"] = self.width_loss_weight

        clip_loss_names = []
        for loss_name in self.losses_to_apply:
            if loss_name in ["clip_conv_loss", "clip_mask_loss"]:
                conv_loss = self.loss_mapper[loss_name](
                    sketches, targets, mode)
                for layer in conv_loss.keys():
                    if "normalization" in layer:
                        loss_coeffs[layer] = 0  # include layer 11 in gradnorm but not in final loss
                        losses_dict[layer] = conv_loss[layer]
                    else:
                        layer_w_index = int(re.findall(r'\d+', layer)[0])  # get the layer's number
                        losses_dict[layer] = conv_loss[layer]
                        loss_coeffs[layer] = self.args.clip_conv_layer_weights[layer_w_index]
                        clip_loss_names.append(layer)
            elif loss_name == "width_loss":
                losses_dict[loss_name] = self.loss_mapper[loss_name](widths, renderer.get_strokes_in_canvas_count())
            elif loss_name == "l2":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets).mean()
            elif loss_name == "ratio_loss":
                continue
            else:
                losses_dict[loss_name] = self.loss_mapper[loss_name](sketches, targets, mode).mean()

        losses_dict_original = losses_dict.copy()
        if self.gradnorm:
            if mode == "train":
                if self.width_optim:
                    self.new_weights = compute_grad_norm_losses(losses_dict, renderer.get_width_mlp(),
                                                                renderer.get_mlp())
                else:
                    self.new_weights = compute_grad_norm_losses(losses_dict, renderer.get_mlp(), renderer.get_mlp())
            # if mode is eval, take the norm wieghts of prev step, since we don't have grads here

            for key in losses_dict.keys():
                # losses_dict_copy[key] = losses_dict_copy[key] * self.new_weights[key]
                losses_dict[key] = losses_dict[key] * self.new_weights[key]

        losses_dict_copy = {}  # return the normalised losses before weighting
        for k_ in losses_dict.keys():
            losses_dict_copy[k_] = losses_dict[k_].clone().detach()
        for key in losses_dict.keys():
            # loss = loss + losses_dict[key] * loss_coeffs[key]
            if loss_coeffs[key] == 0:
                losses_dict[key] = losses_dict[key].detach() * loss_coeffs[key]
            else:
                losses_dict[key] = losses_dict[key] * loss_coeffs[key]

        if self.ratio_loss:
            losses_dict["ratio_loss"] = self.loss_mapper["ratio_loss"](losses_dict_original, clip_loss_names).mean()

        losses_dict_original_detach = {}
        for k_ in losses_dict_original.keys():
            losses_dict_original_detach[k_] = losses_dict_original[k_].clone().detach()

        return losses_dict, losses_dict_copy, losses_dict_original_detach


class CLIPLoss(torch.nn.Module):
    def __init__(self, args, device):
        super(CLIPLoss, self).__init__()

        self.args = args
        self.device = device
        self.model, clip_preprocess = clip.load(
            'ViT-B/32', self.device, jit=False)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [clip_preprocess.transforms[-1]])  # clip normalisation
        self.NUM_AUGS = args.num_aug_clip
        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.calc_target = True
        self.include_target_in_aug = args.include_target_in_aug
        self.counter = 0
        self.augment_both = args.augment_both

    def forward(self, sketches, targets, mode="train"):
        if self.calc_target:
            targets_ = self.preprocess(targets).to(self.device)
            self.targets_features = self.model.encode_image(targets_).detach()
            self.calc_target = False

        if mode == "eval":
            # for regular clip distance, no augmentations
            with torch.no_grad():
                sketches = self.preprocess(sketches).to(self.device)
                sketches_features = self.model.encode_image(sketches)
                return 1. - torch.cosine_similarity(sketches_features, self.targets_features)

        loss_clip = 0
        sketch_augs = []
        img_augs = []
        for n in range(self.NUM_AUGS):
            augmented_pair = self.augment_trans(torch.cat([sketches, targets]))
            sketch_augs.append(augmented_pair[0].unsqueeze(0))

        sketch_batch = torch.cat(sketch_augs)
        sketch_features = self.model.encode_image(sketch_batch)
        for n in range(self.NUM_AUGS):
            loss_clip += (1. - torch.cosine_similarity(
                sketch_features[n:n + 1], self.targets_features, dim=1))
        self.counter += 1
        return loss_clip
        # return 1. - torch.cosine_similarity(sketches_features, self.targets_features)


class LPIPS(torch.nn.Module):
    def __init__(self, pretrained=True, normalize=True, pre_relu=True, device=None):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(LPIPS, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        self.normalize = normalize
        self.pretrained = pretrained
        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        self.augment_trans = transforms.Compose(augemntations)
        self.feature_extractor = LPIPS._FeatureExtractor(
            pretrained, pre_relu).to(device)

    def _l2_normalize_features(self, x, eps=1e-10):
        nrm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        return x / (nrm + eps)

    def forward(self, pred, target, mode="train"):
        """Compare VGG features of two inputs."""

        # Get VGG features

        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0)
        ys = torch.cat(img_augs, dim=0)

        pred = self.feature_extractor(xs)
        target = self.feature_extractor(ys)

        # L2 normalize features
        if self.normalize:
            pred = [self._l2_normalize_features(f) for f in pred]
            target = [self._l2_normalize_features(f) for f in target]

        # TODO(mgharbi) Apply Richard's linear weights?

        if self.normalize:
            diffs = [torch.sum((p - t) ** 2, 1)
                     for (p, t) in zip(pred, target)]
        else:
            # mean instead of sum to avoid super high range
            diffs = [torch.mean((p - t) ** 2, 1)
                     for (p, t) in zip(pred, target)]

        # Spatial average
        diffs = [diff.mean([1, 2]) for diff in diffs]

        return sum(diffs)

    class _FeatureExtractor(torch.nn.Module):
        def __init__(self, pretrained, pre_relu):
            super(LPIPS._FeatureExtractor, self).__init__()
            vgg_pretrained = models.vgg16(pretrained=pretrained).features

            self.breakpoints = [0, 4, 9, 16, 23, 30]
            if pre_relu:
                for i, _ in enumerate(self.breakpoints[1:]):
                    self.breakpoints[i + 1] -= 1

            # Split at the maxpools
            for i, b in enumerate(self.breakpoints[:-1]):
                ops = torch.nn.Sequential()
                for idx in range(b, self.breakpoints[i + 1]):
                    op = vgg_pretrained[idx]
                    ops.add_module(str(idx), op)
                # print(ops)
                self.add_module("group{}".format(i), ops)

            # No gradients
            for p in self.parameters():
                p.requires_grad = False

            # Torchvision's normalization: <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>
            self.register_buffer("shift", torch.Tensor(
                [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("scale", torch.Tensor(
                [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            feats = []
            x = (x - self.shift) / self.scale
            for idx in range(len(self.breakpoints) - 1):
                m = getattr(self, "group{}".format(idx))
                x = m(x)
                feats.append(x)
            return feats


class WidthLoss(torch.nn.Module):
    def __init__(self, args, device):
        super(WidthLoss, self).__init__()
        self.width_loss_type = args.width_loss_type
        self.width_loss_weight = args.width_loss_weight
        self.zero = torch.tensor(0).to(device)

    def forward(self, widths, strokes_in_canvas_count):
        sum_w = torch.sum(widths)
        if self.width_loss_type == "L1_hinge":  # this option is deprecated
            return torch.max(self.zero, sum_w - self.width_loss_weight)
        return sum_w / strokes_in_canvas_count


class RatioLoss(torch.nn.Module):
    def __init__(self, args, device):
        super(RatioLoss, self).__init__()
        self.target_ratio = args.ratio_loss
        self.mse_loss = nn.MSELoss()

    def forward(self, losses_dict_original, clip_loss_names):
        loss_clip = 0
        for clip_loss in clip_loss_names:
            loss_clip = loss_clip + losses_dict_original[clip_loss]
        loss_clip = loss_clip * self.target_ratio
        width_loss = losses_dict_original["width_loss"]
        return self.mse_loss(width_loss, loss_clip)


class L2_(torch.nn.Module):
    def __init__(self):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(L2_, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)
        # LOG.warning("LPIPS is untested")

    def forward(self, pred, target, mode="train"):
        """Compare VGG features of two inputs."""

        # Get VGG features

        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        pred = torch.cat(sketch_augs, dim=0)
        target = torch.cat(img_augs, dim=0)
        diffs = [torch.square(p - t).mean() for (p, t) in zip(pred, target)]
        return sum(diffs)


class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model, device, mask_cls="none", apply_mask=False, mask_attention=False):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None
        self.device = device
        self.n_channels = 3
        self.kernel_h = 32
        self.kernel_w = 32
        self.step = 32
        self.num_patches = 49
        self.mask_cls = mask_cls
        self.apply_mask = apply_mask
        self.mask_attention = mask_attention

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x, masks=None, mode="train"):
        masks_flat = torch.ones((x.shape[0], 50, 768)).to(self.device)  # without any effect
        attn_map = None
        if masks is not None and self.apply_mask:
            x_copy = x.detach().clone()

            patches_x = x_copy.unfold(2, self.kernel_h, self.step).unfold(3, self.kernel_w, self.step).reshape(-1,
                                                                                                               self.n_channels,
                                                                                                               self.num_patches,
                                                                                                               32, 32)
            # split the masks into patches (the same input patches to the transformer)
            # shape is (batch_size, channel, num_patches, patch_size, patch_size) = (5, 3, 49, 32, 32)
            patches_mask = masks.unfold(2, self.kernel_h, self.step).unfold(3, self.kernel_w, self.step).reshape(-1,
                                                                                                                 self.n_channels,
                                                                                                                 self.num_patches,
                                                                                                                 32, 32)
            # masks_ is a binary mask (batch_size, 1, 7, ,7) to say which patch should be masked out
            masks_ = torch.ones((x.shape[0], 1, 7, 7)).to(self.device)
            for i in range(masks.shape[0]):
                for j in range(self.num_patches):
                    # we mask a patch if more than 20% of the patch is masked
                    zeros = (patches_mask[i, 0, j] == 0).sum() / (self.kernel_w * self.kernel_h)
                    if zeros > 0.2:
                        masks_[i, :, j // 7, j % 7] = 0

            if self.mask_attention:
                mask2 = masks_[:, 0].reshape(-1, 49).to(self.device)  # .to(device) shape (5, 49)
                mask2 = torch.cat([torch.ones(mask2.shape[0], 1).to(self.device), mask2], dim=-1)
                mask2 = mask2.unsqueeze(1)
                attn_map = mask2.repeat(1, 50, 1).to(self.device)  # 5, 50, 50
                attn_map[:, 0, 0] = 1
                attn_map = 1 - attn_map
                indixes = (attn_map == 0).nonzero()  # shape [136, 2] [[aug_im],[index]]
                attn_map = attn_map.repeat(12, 1, 1).bool()  # [60, 50, 50]

            # masks_flat's shape is (5, 49), for each image in the batch we have 49 flags indicating if to mask the i'th patch or not
            masks_flat = masks_[:, 0].reshape(-1, self.num_patches)

            # now we add the cls token mask, it's all ones for now since we want to leave it
            # now the shape is (5, 50) where the first number in each of the 5 rows is 1 (meaning - son't mask the cls token)
            masks_flat = torch.cat([torch.ones(masks_flat.shape[0], 1).to(self.device), masks_flat],
                                   dim=1)  # include cls by default
            # now we duplicate this from (5, 50) to (5, 50, 768) to match the tokens dimentions
            masks_flat = masks_flat.unsqueeze(2).repeat(1, 1, 768)  # shape is (5, 50, 768)

        elif self.mask_cls != "none":
            if self.mask_cls == "only_cls":
                masks_flat = torch.zeros((5, 50, 768)).to(self.device)
                masks_flat[:, 0, :] = 1
            elif self.mask_cls == "cls_out":
                masks_flat[:, 0, :] = 0

        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] * masks_flat for k in range(12)]

        return fc_features, featuremaps


def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    if "RN" in clip_model_name:
        return [torch.square(x_conv, y_conv, dim=1).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


class CLIPConvLoss(torch.nn.Module):
    def __init__(self, args, mask, device):
        # mask is a binary tensor with shape (1,3,224,224)
        super(CLIPConvLoss, self).__init__()
        self.device = device

        self.mask = mask
        self.loss_mask = args.loss_mask
        assert self.loss_mask in ["none", "back", "for"]
        self.apply_mask = (self.loss_mask != "none")
        if self.loss_mask == "for":
            # default for the mask is to mask out the background
            # if mask loss is for it means we want to maskout the foreground
            self.mask = 1 - mask

        self.clip_model_name = args.clip_model_name
        assert self.clip_model_name in [
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "ViT-B/32",
            "ViT-B/16",
        ]

        self.clip_conv_loss_type = args.clip_conv_loss_type
        self.clip_fc_loss_type = "Cos"  # args.clip_fc_loss_type
        assert self.clip_conv_loss_type in [
            "L2", "Cos", "L1",
        ]
        assert self.clip_fc_loss_type in [
            "L2", "Cos", "L1",
        ]

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, clip_preprocess = clip.load(
            self.clip_model_name, self.device, jit=False)

        if self.clip_model_name.startswith("ViT"):
            self.loss_log_name = "vit"
            self.visual_encoder = CLIPVisualEncoder(self.model, self.device)
            self.l11_norm = False

        else:
            self.loss_log_name = "rn"
            self.visual_model = self.model.visual
            layers = list(self.model.visual.children())
            init_layers = torch.nn.Sequential(*layers)[:8]
            self.layer1 = layers[8]
            self.layer2 = layers[9]
            self.layer3 = layers[10]
            self.layer4 = layers[11]
            self.att_pool2d = layers[12]

        self.args = args

        self.img_size = clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()

        self.num_augs = self.args.num_aug_clip

        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.clip_fc_layer_dims = None  # self.args.clip_fc_layer_dims
        self.clip_conv_layer_dims = None  # self.args.clip_conv_layer_dims
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.counter = 0

    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        conv_loss_dict = {}
        if self.apply_mask:
            sketch *= self.mask

        x = sketch.to(self.device)
        y = target.to(self.device)

        sketch_augs, img_augs = [self.normalize_transform(x)], [
            self.normalize_transform(y)]
        if mode == "train":
            for n in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([x, y]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        if self.clip_model_name.startswith("RN"):
            xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(
                xs.contiguous())
            ys_fc_features, ys_conv_features = self.forward_inspection_clip_resnet(
                ys.detach())

        else:
            xs_fc_features, xs_conv_features = self.visual_encoder(xs, mode=mode)
            ys_fc_features, ys_conv_features = self.visual_encoder(ys, mode=mode)

        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, self.clip_model_name)

        for layer, w in enumerate(self.args.clip_conv_layer_weights):
            if w:
                conv_loss_dict[f"clip_{self.loss_log_name}_l{layer}"] = conv_loss[layer]
            if layer == 11 and self.l11_norm:
                conv_loss_dict[f"clip_{self.loss_log_name}_l{layer}_normalization"] = conv_loss[layer]

        if self.clip_fc_loss_weight:
            # fc distance is always cos
            # fc_loss = torch.nn.functional.mse_loss(xs_fc_features, ys_fc_features).mean()
            fc_loss = (1 - torch.cosine_similarity(xs_fc_features,
                                                   ys_fc_features, dim=1)).mean()
            conv_loss_dict[f"fc_{self.loss_log_name}"] = fc_loss * self.clip_fc_loss_weight

        self.counter += 1
        return conv_loss_dict

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x

        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]


class CLIPmaskLoss(torch.nn.Module):
    def __init__(self, args, mask, device):
        super(CLIPmaskLoss, self).__init__()
        self.args = args
        self.mask = mask
        self.device = device
        self.loss_mask = args.loss_mask
        assert self.loss_mask in ["none", "back", "for", "back_latent", "for_latent"]
        self.apply_mask = (self.loss_mask != "none")
        self.dilated_mask = args.dilated_mask
        if self.dilated_mask:
            kernel_tensor = torch.ones((1, 1, 11, 11)).to(self.device)
            mask_ = torch.clamp(
                torch.nn.functional.conv2d(mask[:, 0, :, :].unsqueeze(1), kernel_tensor, padding=(5, 5)), 0, 1)
            mask = torch.cat([mask_, mask_, mask_], axis=1)

        if "for" in self.loss_mask:
            # default for the mask is to mask out the background
            # if mask loss is for it means we want to maskout the foreground
            self.mask = 1 - mask

        self.clip_model_name = args.clip_model_name
        self.clip_for_model_name = "RN101"
        self.valid_models = [
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "ViT-B/32",
            "ViT-B/16",
        ]
        assert self.clip_model_name in self.valid_models and self.clip_for_model_name in self.valid_models

        self.clip_conv_layer_weights = args.clip_conv_layer_weights
        self.clip_conv_loss_type = args.clip_conv_loss_type
        self.clip_fc_loss_type = "Cos"
        self.num_augs = args.num_aug_clip

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        # background model (ViT)
        self.model, clip_preprocess = clip.load(
            self.clip_model_name, self.device, jit=False)
        self.model.eval()
        if self.clip_model_name.startswith("ViT"):
            self.visual_encoder = CLIPVisualEncoder(self.model, self.device, args.mask_cls, self.apply_mask,
                                                    args.mask_attention)

        self.img_size = clip_preprocess.transforms[1].size

        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            # clip_preprocess.transforms[0],  # Resize
            # clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        # augemntations.append(transforms.RandomResizedCrop(
        #     224, scale=(0.4, 0.9), ratio=(1.0, 1.0)))

        self.augment_trans = transforms.Compose(augemntations)
        self.clip_fc_layer_dims = None  # self.args.clip_fc_layer_dims
        self.clip_conv_layer_dims = None  # self.args.clip_conv_layer_dims
        self.clip_fc_loss_weight = 0
        self.counter = 0

    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        conv_loss_dict = {}

        x = sketch.to(self.device)
        y = target.to(self.device)
        sketch_augs, img_augs, masks = [x], [y], [self.mask]
        if mode == "train":
            for n in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([x, y, self.mask]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))
                masks.append(augmented_pair[2].unsqueeze(0))
        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)
        masks = torch.cat(masks, dim=0).to(self.device)
        masks[masks < 0.5] = 0
        masks[masks >= 0.5] = 1
        # background pass
        if self.apply_mask and "latent" not in self.loss_mask:
            # if "latent" not in self.loss_mask:
            xs_back = self.normalize_transform(xs * masks)
        else:
            xs_back = self.normalize_transform(xs)
        ys_back = self.normalize_transform(ys)
        if "latent" not in self.loss_mask:
            masks = None
        xs_fc_features, xs_conv_features = self.visual_encoder(xs_back, masks, mode=mode)
        ys_fc_features, ys_conv_features = self.visual_encoder(ys_back, masks, mode=mode)
        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, self.clip_model_name)
        for layer, w in enumerate(self.clip_conv_layer_weights):
            if w:
                conv_loss_dict[f"clip_vit_l{layer}"] = conv_loss[layer] * w

        self.counter += 1
        return conv_loss_dict

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x

        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]
