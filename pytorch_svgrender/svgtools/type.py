# -*- coding: utf-8 -*-
# Author: ximing
# Description: SVGDreamer - type checking
# Copyright (c) 2023, XiMing Xing.
# License: MIT License

from typing import AnyStr

import xml.etree.ElementTree as ET


def is_valid_svg(file_path: AnyStr) -> bool:
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        if root.tag.endswith('svg') and 'xmlns' in root.attrib:
            return True
        else:
            return False
    except ET.ParseError:
        return False
