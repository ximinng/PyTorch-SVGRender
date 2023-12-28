# -*- coding: utf-8 -*-
# Author: ximing
# Description: SVGDreamer - optim
# Copyright (c) 2023, XiMing Xing.
# License: MIT License
from functools import partial

import torch
from omegaconf import DictConfig


def get_optimizer(optimizer_name, parameters, lr=None, config: DictConfig = None):
    param_dict = {}
    if optimizer_name == "adam":
        optimizer = partial(torch.optim.Adam, params=parameters)
        if lr is not None:
            optimizer = partial(torch.optim.Adam, params=parameters, lr=lr)
        if config.get('betas'):
            param_dict['betas'] = config.betas
        if config.get('weight_decay'):
            param_dict['weight_decay'] = config.weight_decay
        if config.get('eps'):
            param_dict['eps'] = config.eps
    elif optimizer_name == "adamW":
        optimizer = partial(torch.optim.AdamW, params=parameters)
        if lr is not None:
            optimizer = partial(torch.optim.AdamW, params=parameters, lr=lr)
        if config.get('betas'):
            param_dict['betas'] = config.betas
        if config.get('weight_decay'):
            param_dict['weight_decay'] = config.weight_decay
        if config.get('eps'):
            param_dict['eps'] = config.eps
    elif optimizer_name == "radam":
        optimizer = partial(torch.optim.RAdam, params=parameters)
        if lr is not None:
            optimizer = partial(torch.optim.RAdam, params=parameters, lr=lr)
        if config.get('betas'):
            param_dict['betas'] = config.betas
        if config.get('weight_decay'):
            param_dict['weight_decay'] = config.weight_decay
    elif optimizer_name == "sgd":
        optimizer = partial(torch.optim.SGD, params=parameters)
        if lr is not None:
            optimizer = partial(torch.optim.SGD, params=parameters, lr=lr)
        if config.get('momentum'):
            param_dict['momentum'] = config.momentum
        if config.get('weight_decay'):
            param_dict['weight_decay'] = config.weight_decay
        if config.get('nesterov'):
            param_dict['nesterov'] = config.nesterov
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not implemented.")

    if len(param_dict.keys()) > 0:
        return optimizer(**param_dict)
    else:
        return optimizer()
