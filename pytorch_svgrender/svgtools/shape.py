# -*- coding: utf-8 -*-
# Author: ximing
# Description: SVGDreamer - shape
# Copyright (c) 2023, XiMing Xing.
# License: MIT License

import xml.etree.ElementTree as ET


def circle_tag(cx: float, cy: float, r: float, transform: str = None):
    attrib = {
        'cx': f'{cx}', 'cy': f'{cy}', 'r': f'{r}'
    }
    if transform is not None:
        attrib['transform'] = transform
    _circle = ET.Element('circle', attrib)  # tag, attrib
    return _circle


def rect_tag(
        x: float, y: float, rx: float, ry: float,
        width: float = 600, height: float = 600,
        transform: str = None
):
    attrib = {
        'x': f'{x}', 'y': f'{y}', 'rx': f'{rx}', 'ry': f'{ry}',
        'width': f'{width}', 'height': f'{height}'
    }
    if transform is not None:
        attrib['transform'] = transform
    _rect = ET.Element('rect', attrib)  # tag, attrib
    return _rect
