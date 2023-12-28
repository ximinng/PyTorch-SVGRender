# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2023, XiMing Xing.
# License: MIT License

from .tff import FONT_LIST
from .type import is_valid_svg
from .merge import merge_svg_files
from .process import delete_empty_path, add_def_tag

__all__ = [
    'is_valid_svg',
    'merge_svg_files',
    'FONT_LIST',
    'delete_empty_path', 'add_def_tag'
]
