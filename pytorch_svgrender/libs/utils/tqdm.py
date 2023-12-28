# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

from typing import Callable
from tqdm.auto import tqdm


def tqdm_decorator(func: Callable):
    """A decorator function called tqdm_decorator that takes a function as an argument and
    returns a new function that wraps the input function with a tqdm progress bar.

    Noting: **The input function is assumed to have an object self as its first argument**, which contains a step attribute,
    an args attribute with a train_num_steps attribute, and an accelerator attribute with an is_main_process attribute.

    Args:
        func: tqdm_decorator

    Returns:
            a new function that wraps the input function with a tqdm progress bar.
    """

    def wrapper(*args, **kwargs):
        with tqdm(initial=args[0].step,
                  total=args[0].args.train_num_steps,
                  disable=not args[0].accelerator.is_main_process) as pbar:
            func(*args, **kwargs, pbar=pbar)

    return wrapper
