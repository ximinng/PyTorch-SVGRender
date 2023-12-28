# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import math

import torch


def identity(t, *args, **kwargs):
    """return t"""
    return t


def exists(x):
    """whether x is None or not"""
    return x is not None


def default(val, d):
    """ternary judgment: val != None ? val : d"""
    if exists(val):
        return val
    return d() if callable(d) else d


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


#################################################################################
#                             Model Utils                                       #
#################################################################################

def sum_params(model: torch.nn.Module, eps: float = 1e6):
    return sum(p.numel() for p in model.parameters()) / eps


#################################################################################
#                            DataLoader Utils                                   #
#################################################################################

def cycle(dl):
    while True:
        for data in dl:
            yield data


#################################################################################
#                            Diffusion Model Utils                              #
#################################################################################

def extract(a, t, x_shape):
    b, *_ = t.shape
    assert x_shape[0] == b
    out = a.gather(-1, t)  # 1-D tensor, shape: (b,)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # shape: [b, 1, 1, 1]


def unnormalize(x):
    """unnormalize_to_zero_to_one"""
    x = (x + 1) * 0.5  # Map the data interval to [0, 1]
    return torch.clamp(x, 0.0, 1.0)


def normalize(x):
    """normalize_to_neg_one_to_one"""
    x = x * 2 - 1  # Map the data interval to [-1, 1]
    return torch.clamp(x, -1.0, 1.0)
