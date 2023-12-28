# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description: a self consistent system,
# including runner, trainer, loss function, EMA, optimizer, lr scheduler , and common utils.

from .utils import lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules={'engine', 'metric', 'modules', 'solver', 'utils'},
    submod_attrs={}
)

__version__ = '0.0.1'
