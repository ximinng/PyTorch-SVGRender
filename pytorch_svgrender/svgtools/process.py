# -*- coding: utf-8 -*-
# Author: ximing
# Description: process
# Copyright (c) 2023, XiMing Xing.
# License: MIT License

import xml.etree.ElementTree as ET
from typing import Tuple

import omegaconf

from .shape import circle_tag, rect_tag
from .type import is_valid_svg

def delete_empty_path(input_svg: str, output_svg: str):
    is_valid_svg(input_svg)

    # read svg
    tree = ET.parse(input_svg)
    root = tree.getroot()

    group = ET.Element('g')
    for i, element in enumerate(root.iter()):
        element.tag = element.tag.split('}')[-1]
        if element.tag == 'path':
            if element.get('d') == 'C  NaN NaN' or element.get('d') == '':
                continue
            group.append(element)

    # new svg
    svg = ET.Element('svg',
                     xmlns="http://www.w3.org/2000/svg",
                     version='1.1',
                     width=root.get('width'),
                     height=root.get('height'),
                     viewBox=root.get('viewBox'))
    svg.append(group)
    tree = ET.ElementTree(svg)
    tree.write(output_svg, encoding='utf-8', xml_declaration=True)


def add_clipPath2def(mounted_node: ET.Element, tag_name: str, attrs: omegaconf.DictConfig):
    # add defs node
    defs = ET.SubElement(mounted_node, 'defs')  # parent=mounted_node, tag='defs'
    if tag_name == 'none':
        return None
    # add clipPath node
    id = 'def_clip'
    _circleClip = ET.SubElement(defs, 'clipPath', id='def_clip')  # parent=defs, tag='clipPath'
    # add ops
    if tag_name == 'circle_clip':
        _circleClip.append(
            circle_tag(cx=attrs.cx, cy=attrs.cy, r=attrs.r)
        )
    elif tag_name == 'rect_clip':
        _circleClip.append(
            rect_tag(x=attrs.x, y=attrs.y, rx=attrs.rx, ry=attrs.ry, width=attrs.width, height=attrs.height)
        )
    else:
        raise NotImplementedError(f'{tag_name} is not exist!')
    return id


def add_def_tag(
        svg_path: str,
        def_tag_plan: str,
        out_size: Tuple[int, int],  # e.g.: (600, 600)
):
    is_valid_svg(svg_path)

    width, height = out_size[0], out_size[1]

    # set def tag
    if def_tag_plan == 'circle_clip':
        def_cfg = omegaconf.DictConfig({
            'name': 'circle_clip',
            'attrs': {'cx': width // 2, 'cy': height // 2, 'r': int(height * 0.5)}
        })
    elif def_tag_plan == 'rect_clip':
        def_cfg = omegaconf.DictConfig({
            'name': 'rect_clip',
            'attrs': {'x': 0, 'y': 0, 'rx': 70, 'ry': 70, 'width': width, 'height': height}
        })
    else:
        def_cfg = None

    # load SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()
    # new group, and add paths form svg_path_1
    group = ET.Element('g')
    for i, element in enumerate(root.iter()):
        element.tag = element.tag.split('}')[-1]
        if element.tag in ['path', 'polygon']:
            group.append(element)

    # new svg
    svg = ET.Element('svg',
                     xmlns="http://www.w3.org/2000/svg",
                     version='1.1',
                     width=str(out_size[0]),
                     height=str(out_size[1]))
    # add def tag to the SVG
    clip_id = add_clipPath2def(mounted_node=svg,
                               tag_name=def_cfg.name,
                               attrs=def_cfg.attrs)
    group.set('clip-path', f'url(#{clip_id})')
    svg.append(group)
    # write svg
    tree = ET.ElementTree(svg)
    tree.write(svg_path, encoding='utf-8', xml_declaration=True)
