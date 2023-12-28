# -*- coding: utf-8 -*-
# Author: ximing
# Description: parent class
# Copyright (c) 2023, XiMing Xing.
# License: MIT License
import pathlib
from typing import AnyStr, List, Union
import xml.etree.ElementTree as etree

import torch
import pydiffvg


def init_pydiffvg(device: torch.device,
                  use_gpu: bool = torch.cuda.is_available(),
                  print_timing: bool = False):
    pydiffvg.set_use_gpu(use_gpu)
    pydiffvg.set_device(device)
    pydiffvg.set_print_timing(print_timing)


class DiffVGState(torch.nn.Module):

    def __init__(self,
                 device: torch.device,
                 use_gpu: bool = torch.cuda.is_available(),
                 print_timing: bool = False,
                 canvas_width: int = None,
                 canvas_height: int = None):
        super(DiffVGState, self).__init__()
        # pydiffvg device setting
        self.device = device
        init_pydiffvg(device, use_gpu, print_timing)

        # canvas size
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        # record all paths
        self.shapes = []
        self.shape_groups = []
        # record the current optimized path
        self.cur_shapes = []
        self.cur_shape_groups = []

        # learnable SVG params
        self.point_vars = []
        self.color_vars = []
        self.width_vars = []

    def clip_curve_shape(self, *args, **kwargs):
        raise NotImplementedError

    def render_warp(self, seed=0):
        self.clip_curve_shape()

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups
        )
        _render = pydiffvg.RenderFunction.apply
        img = _render(self.canvas_width,  # width
                      self.canvas_height,  # height
                      2,  # num_samples_x
                      2,  # num_samples_y
                      seed,  # seed
                      None,
                      *scene_args)
        return img

    @staticmethod
    def load_svg(path_svg):
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(path_svg)
        return canvas_width, canvas_height, shapes, shape_groups

    def save_svg(self,
                 filename: Union[AnyStr, pathlib.Path],
                 width: int = None,
                 height: int = None,
                 shapes: List = None,
                 shape_groups: List = None,
                 use_gamma: bool = False,
                 background: str = None):
        """
        Save an SVG file with specified parameters and shapes.
        Noting: New version of SVG saving function that is an adaptation of pydiffvg.save_svg.
        The original version saved words resulting in incomplete glyphs.

        Args:
            filename (str): The path to save the SVG file.
            width (int): The width of the SVG canvas.
            height (int): The height of the SVG canvas.
            shapes (list): A list of shapes to be included in the SVG.
            shape_groups (list): A list of shape groups.
            use_gamma (bool): Flag indicating whether to apply gamma correction.
            background (str, optional): The background color of the SVG.

        Returns:
            None
        """
        root = etree.Element('svg')
        root.set('version', '1.1')
        root.set('xmlns', 'http://www.w3.org/2000/svg')
        root.set('width', str(width))
        root.set('height', str(height))

        if background is not None:
            print(f"setting background to {background}")
            root.set('style', str(background))

        defs = etree.SubElement(root, 'defs')
        g = etree.SubElement(root, 'g')

        if use_gamma:
            f = etree.SubElement(defs, 'filter')
            f.set('id', 'gamma')
            f.set('x', '0')
            f.set('y', '0')
            f.set('width', '100%')
            f.set('height', '100%')
            gamma = etree.SubElement(f, 'feComponentTransfer')
            gamma.set('color-interpolation-filters', 'sRGB')
            feFuncR = etree.SubElement(gamma, 'feFuncR')
            feFuncR.set('type', 'gamma')
            feFuncR.set('amplitude', str(1))
            feFuncR.set('exponent', str(1 / 2.2))
            feFuncG = etree.SubElement(gamma, 'feFuncG')
            feFuncG.set('type', 'gamma')
            feFuncG.set('amplitude', str(1))
            feFuncG.set('exponent', str(1 / 2.2))
            feFuncB = etree.SubElement(gamma, 'feFuncB')
            feFuncB.set('type', 'gamma')
            feFuncB.set('amplitude', str(1))
            feFuncB.set('exponent', str(1 / 2.2))
            feFuncA = etree.SubElement(gamma, 'feFuncA')
            feFuncA.set('type', 'gamma')
            feFuncA.set('amplitude', str(1))
            feFuncA.set('exponent', str(1 / 2.2))
            g.set('style', 'filter:url(#gamma)')

        # Store color
        for i, shape_group in enumerate(shape_groups):
            def add_color(shape_color, name):
                if isinstance(shape_color, pydiffvg.LinearGradient):
                    lg = shape_color
                    color = etree.SubElement(defs, 'linearGradient')
                    color.set('id', name)
                    color.set('x1', str(lg.begin[0].item()))
                    color.set('y1', str(lg.begin[1].item()))
                    color.set('x2', str(lg.end[0].item()))
                    color.set('y2', str(lg.end[1].item()))
                    offsets = lg.offsets.data.cpu().numpy()
                    stop_colors = lg.stop_colors.data.cpu().numpy()
                    for j in range(offsets.shape[0]):
                        stop = etree.SubElement(color, 'stop')
                        stop.set('offset', str(offsets[j]))
                        c = lg.stop_colors[j, :]
                        stop.set('stop-color', 'rgb({}, {}, {})'.format(
                            int(255 * c[0]), int(255 * c[1]), int(255 * c[2])
                        ))
                        stop.set('stop-opacity', '{}'.format(c[3]))
                if isinstance(shape_color, pydiffvg.RadialGradient):
                    lg = shape_color
                    color = etree.SubElement(defs, 'radialGradient')
                    color.set('id', name)
                    color.set('cx', str(lg.center[0].item() / width))
                    color.set('cy', str(lg.center[1].item() / height))
                    # this only support width=height
                    color.set('r', str(lg.radius[0].item() / width))
                    offsets = lg.offsets.data.cpu().numpy()
                    stop_colors = lg.stop_colors.data.cpu().numpy()
                    for j in range(offsets.shape[0]):
                        stop = etree.SubElement(color, 'stop')
                        stop.set('offset', str(offsets[j]))
                        c = lg.stop_colors[j, :]
                        stop.set('stop-color', 'rgb({}, {}, {})'.format(
                            int(255 * c[0]), int(255 * c[1]), int(255 * c[2])
                        ))
                        stop.set('stop-opacity', '{}'.format(c[3]))

            if shape_group.fill_color is not None:
                add_color(shape_group.fill_color, 'shape_{}_fill'.format(i))
            if shape_group.stroke_color is not None:
                add_color(shape_group.stroke_color, 'shape_{}_stroke'.format(i))

        for i, shape_group in enumerate(shape_groups):
            shape = shapes[shape_group.shape_ids[0]]
            if isinstance(shape, pydiffvg.Circle):
                shape_node = etree.SubElement(g, 'circle')
                shape_node.set('r', str(shape.radius.item()))
                shape_node.set('cx', str(shape.center[0].item()))
                shape_node.set('cy', str(shape.center[1].item()))
            elif isinstance(shape, pydiffvg.Polygon):
                shape_node = etree.SubElement(g, 'polygon')
                points = shape.points.data.cpu().numpy()
                path_str = ''
                for j in range(0, shape.points.shape[0]):
                    path_str += '{} {}'.format(points[j, 0], points[j, 1])
                    if j != shape.points.shape[0] - 1:
                        path_str += ' '
                shape_node.set('points', path_str)
            elif isinstance(shape, pydiffvg.Path):
                for j, id in enumerate(shape_group.shape_ids):
                    shape = shapes[id]
                    if isinstance(shape, pydiffvg.Path):
                        if j == 0:
                            shape_node = etree.SubElement(g, 'path')
                            node_id = shape_node.get('id')
                            path_str = ''

                        num_segments = shape.num_control_points.shape[0]
                        num_control_points = shape.num_control_points.data.cpu().numpy()
                        points = shape.points.data.cpu().numpy()
                        num_points = shape.points.shape[0]
                        path_str += 'M {} {}'.format(points[0, 0], points[0, 1])
                        point_id = 1
                        for j in range(0, num_segments):
                            if num_control_points[j] == 0:
                                p = point_id % num_points
                                path_str += ' L {} {}'.format(
                                    points[p, 0], points[p, 1])
                                point_id += 1
                            elif num_control_points[j] == 1:
                                p1 = (point_id + 1) % num_points
                                path_str += ' Q {} {} {} {}'.format(
                                    points[point_id, 0], points[point_id, 1],
                                    points[p1, 0], points[p1, 1])
                                point_id += 2
                            elif num_control_points[j] == 2:
                                p2 = (point_id + 2) % num_points
                                path_str += ' C {} {} {} {} {} {}'.format(
                                    points[point_id, 0], points[point_id, 1],
                                    points[point_id + 1, 0], points[point_id + 1, 1],
                                    points[p2, 0], points[p2, 1])
                                point_id += 3
                if node_id is not None:
                    shape_node.set('id', node_id)  # add id to Path
                shape_node.set('d', path_str)
            elif isinstance(shape, pydiffvg.Rect):
                shape_node = etree.SubElement(g, 'rect')
                shape_node.set('x', str(shape.p_min[0].item()))
                shape_node.set('y', str(shape.p_min[1].item()))
                shape_node.set('width', str(shape.p_max[0].item() - shape.p_min[0].item()))
                shape_node.set('height', str(shape.p_max[1].item() - shape.p_min[1].item()))
            elif isinstance(shape, pydiffvg.Ellipse):
                shape_node = etree.SubElement(g, 'ellipse')
                shape_node.set('cx', str(shape.center[0].item()))
                shape_node.set('cy', str(shape.center[1].item()))
                shape_node.set('rx', str(shape.radius[0].item()))
                shape_node.set('ry', str(shape.radius[1].item()))
            else:
                raise NotImplementedError(f'shape type: {type(shape)} is not involved in pydiffvg.')

            shape_node.set('stroke-width', str(2 * shape.stroke_width.data.cpu().item()))
            if shape_group.fill_color is not None:
                if isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                    shape_node.set('fill', 'url(#shape_{}_fill)'.format(i))
                else:
                    c = shape_group.fill_color.data.cpu().numpy()
                    shape_node.set('fill', 'rgb({}, {}, {})'.format(
                        int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                    shape_node.set('opacity', str(c[3]))
            else:
                shape_node.set('fill', 'none')
            if shape_group.stroke_color is not None:
                if isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
                    shape_node.set('stroke', 'url(#shape_{}_stroke)'.format(i))
                else:
                    c = shape_group.stroke_color.data.cpu().numpy()
                    shape_node.set('stroke', 'rgb({}, {}, {})'.format(
                        int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                    shape_node.set('stroke-opacity', str(c[3]))
                shape_node.set('stroke-linecap', 'round')
                shape_node.set('stroke-linejoin', 'round')

        with open(filename, "w") as f:
            f.write(pydiffvg.prettify(root))

    @staticmethod
    def save_image(img, filename, gamma=1):
        if torch.is_tensor(img) and torch.device != 'cpu':
            img = img.detach().cpu()
        pydiffvg.imwrite(img, filename, gamma=gamma)
