# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
from . import lazy

# __getattr__, __dir__, __all__ = lazy.attach(
#     __name__,
#     submodules={},
#     submod_attrs={
#         'misc': ['identity', 'exists', 'default', 'has_int_squareroot', 'sum_params', 'cycle', 'num_to_groups',
#                  'extract', 'normalize', 'unnormalize'],
#         'tqdm': ['tqdm_decorator'],
#         'lazy': ['load']
#     }
# )

from .misc import (
    identity,
    exists,
    default,
    has_int_squareroot,
    sum_params,
    cycle,
    num_to_groups,
    extract,
    normalize,
    unnormalize
)
from .tqdm import tqdm_decorator
