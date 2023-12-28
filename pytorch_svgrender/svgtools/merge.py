# -*- coding: utf-8 -*-
# Author: ximing
# Description: SVGDreamer - merge
# Copyright (c) 2023, XiMing Xing.
# License: MIT License
from typing import Tuple, AnyStr

import omegaconf
from svgpathtools import svg2paths, wsvg

from .type import is_valid_svg
from .shape import *


def merge_svg_files(
        svg_path_1: AnyStr,
        svg_path_2: AnyStr,
        merge_type: str,
        output_svg_path: AnyStr,
        out_size: Tuple[int, int],  # e.g.: (600, 600)
):
    is_valid_svg(svg_path_1)
    is_valid_svg(svg_path_2)

    # set merge ops
    if merge_type.startswith('vert'):  # Move up/down vertically
        if '+' in merge_type:  # move up
            move_val = merge_type.split("+")[1]
            move_val = int(move_val)
        elif '-' in merge_type:  # move down
            move_val = merge_type.split("-")[1]
            move_val = -int(move_val)
        else:
            raise NotImplemented(f'{merge_type} is invalid.')

        merge_svg_by_group(svg_path_1, svg_path_2,
                           cp_offset=(0, move_val),
                           svg_out=output_svg_path, out_size=out_size)

    elif merge_type.startswith('cp'):  # Move all control points
        if '+' in merge_type:
            move_val = merge_type.split("+")[1]
            move_val = int(move_val)
        elif '-' in merge_type:
            move_val = merge_type.split("-")[1]
            move_val = -int(move_val)
        else:
            raise NotImplemented(f'{merge_type} is invalid.')

        merge_svg_by_cp(svg_path_1, svg_path_2,
                        p_offset=move_val,
                        svg_out=output_svg_path, out_size=out_size)

    elif merge_type == 'simple':  # simply combine two SVG files
        simple_merge(svg_path_1, svg_path_2, output_svg_path, out_size)
    else:
        raise NotImplemented(f'{str(merge_type)} is not support !')


def simple_merge(svg_path1, svg_path2, output_path, out_size):
    # read svg to paths
    paths1, attributes1 = svg2paths(svg_path1)
    paths2, attributes2 = svg2paths(svg_path2)
    # merge path and attributes
    paths = paths1 + paths2
    attributes = attributes1 + attributes2
    # write merged svg
    wsvg(paths,
         attributes=attributes,
         filename=output_path,
         viewbox=f"0 0 {out_size[0]} {out_size[1]}")


def merge_svg_by_group(
        svg_path_1: AnyStr,
        svg_path_2: AnyStr,
        cp_offset: Tuple[float, float],
        svg_out: AnyStr,
        out_size: Tuple[int, int],  # e.g.: (600, 600)
):
    # load svg_path_1
    tree1 = ET.parse(svg_path_1)
    root1 = tree1.getroot()
    # new group, and add paths form svg_path_1
    group1 = ET.Element('g')
    for i, element in enumerate(root1.iter()):
        element.tag = element.tag.split('}')[-1]
        if element.tag in ['path', 'polygon']:
            group1.append(element)

    # load svg_path_2
    tree2 = ET.parse(svg_path_2)
    root2 = tree2.getroot()
    # new group, and add paths form svg_path_2
    group2 = ET.Element('g')
    for j, path in enumerate(root2.findall('.//{http://www.w3.org/2000/svg}path')):
        # Remove the 'svg:' prefix from the tag name
        path.tag = path.tag.split('}')[-1]
        group2.append(path)

    # new svg
    svg = ET.Element('svg',
                     xmlns="http://www.w3.org/2000/svg",
                     version='1.1',
                     width=str(out_size[0]),
                     height=str(out_size[1]))

    # control group2
    if 'transform' in group2.attrib:
        group2.attrib['transform'] += f' translate({cp_offset[0]}, {cp_offset[1]})'
    else:
        group2.attrib['transform'] = f'translate({cp_offset[0]}, {cp_offset[1]})'
    # add two group
    svg.append(group1)
    svg.append(group2)
    # write svg
    tree = ET.ElementTree(svg)
    tree.write(svg_out, encoding='utf-8', xml_declaration=True)


def merge_svg_by_cp(
        svg_path_1: AnyStr,
        svg_path_2: AnyStr,
        p_offset: float,
        svg_out: AnyStr,
        out_size: Tuple[int, int],  # e.g.: (600, 600)
):
    # load svg_path_1
    tree1 = ET.parse(svg_path_1)
    root1 = tree1.getroot()
    # new group, and add paths form svg_path_1
    group1 = ET.Element('g')
    for i, element in enumerate(root1.iter()):
        element.tag = element.tag.split('}')[-1]
        if element.tag in ['path', 'polygon']:
            group1.append(element)

    # load svg_path_2
    tree2 = ET.parse(svg_path_2)
    root2 = tree2.getroot()

    # new group, and add paths form svg_path_2
    group2 = ET.Element('g')
    for j, path in enumerate(root2.findall('.//{http://www.w3.org/2000/svg}path')):
        # remove the 'svg:' prefix from the tag name
        path.tag = path.tag.split('}')[-1]

        d = path.get('d')
        # parse paths
        path_data = d.split()
        new_path_data = []

        for i in range(len(path_data)):
            if path_data[i].replace('.', '').isdigit():  # get point coordinates
                new_param = float(path_data[i]) + p_offset
                new_path_data.append(str(new_param))
            else:
                new_path_data.append(path_data[i])
        # update new d attrs
        path.set('d', ' '.join(new_path_data))

        group2.append(path)

    # new svg
    svg = ET.Element('svg',
                     xmlns="http://www.w3.org/2000/svg",
                     version='1.1',
                     width=str(out_size[0]),
                     height=str(out_size[1]))

    # add two group
    svg.append(group1)
    svg.append(group2)
    # write svg
    tree = ET.ElementTree(svg)
    tree.write(svg_out, encoding='utf-8', xml_declaration=True)


def merge_two_svgs_edit(
        svg_path_1: AnyStr,
        svg_path_2: AnyStr,
        def_cfg: omegaconf.DictConfig,
        p2_offset: Tuple[float, float],
        svg_out: AnyStr,
        out_size: Tuple[int, int],  # e.g.: (600, 600)
):
    # load svg_path_1
    tree1 = ET.parse(svg_path_1)
    root1 = tree1.getroot()
    # new group, and add paths form svg_path_1
    group1 = ET.Element('g')
    for i, element in enumerate(root1.iter()):
        element.tag = element.tag.split('}')[-1]
        if element.tag in ['path', 'polygon']:
            group1.append(element)

    # load svg_path_2
    tree2 = ET.parse(svg_path_2)
    root2 = tree2.getroot()

    # new group, and add paths form svg_path_2
    group2 = ET.Element('g')
    for j, path in enumerate(root2.findall('.//{http://www.w3.org/2000/svg}path')):
        # remove the 'svg:' prefix from the tag name
        path.tag = path.tag.split('}')[-1]

        d = path.get('d')
        # parse paths
        path_data = d.split()
        new_path_data = []

        d_idx = 0  # count digit
        for i in range(len(path_data)):
            if path_data[i].replace('.', '').isdigit():  # get point coordinates
                d_idx += 1
                if d_idx % 2 == 1:  # update y
                    new_param = float(path_data[i]) + (p2_offset[1])
                    new_path_data.append(str(new_param))
                else:
                    new_path_data.append(path_data[i])
            else:
                new_path_data.append(path_data[i])
        # update new d attrs
        path.set('d', ' '.join(new_path_data))

        group2.append(path)

    # new svg
    svg = ET.Element('svg',
                     xmlns="http://www.w3.org/2000/svg",
                     version='1.1',
                     width=str(out_size[0]),
                     height=str(out_size[1]))

    # add two group
    svg.append(group1)
    svg.append(group2)
    # write svg
    tree = ET.ElementTree(svg)
    tree.write(svg_out, encoding='utf-8', xml_declaration=True)
