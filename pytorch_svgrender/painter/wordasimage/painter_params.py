import os
import pathlib

import numpy as np
import pydiffvg
import torch
from torch.optim.lr_scheduler import LambdaLR

from pytorch_svgrender.diffvg_warp import DiffVGState
from .ttf import font_string_to_beziers, write_letter_svg


class Painter(DiffVGState):

    def __init__(self,
                 font: str,
                 canvas_size: int,
                 device: torch.device):
        super(Painter, self).__init__(device=device, use_gpu=True, canvas_width=canvas_size, canvas_height=canvas_size)
        self.font = font

    def init_shape(self, path_svg, seed=0):
        assert pathlib.Path(path_svg).exists(), f"{path_svg} is not exist!"
        print(f"-> init svg from `{path_svg}` ...")
        # 1. load svg from path
        canvas_width, canvas_height, self.shapes, self.shape_groups = self.load_svg(path_svg)
        # 2. set learnable parameters
        self.set_point_parameters()

        img = self.render_warp(seed)
        img = img[:, :, 3:4] * img[:, :, :3] + \
              torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_image(self, step: int = 0):
        img = self.render_warp(step)
        img = img[:, :, 3:4] * img[:, :, :3] + \
              torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def clip_curve_shape(self):
        for group in self.shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

    def set_point_parameters(self):  # stroke`s location optimization
        self.point_vars = []
        for i, path in enumerate(self.shapes):
            path.points.requires_grad = True
            self.point_vars.append(path.points)

    def get_point_parameters(self):
        return self.point_vars

    def preprocess_font(self, word, letter, level_of_cc=1, font_path=None, init_path=None):
        if level_of_cc == 0:
            target_cp = None
        else:
            target_cp = {"A": 120, "B": 120, "C": 100, "D": 100,
                         "E": 120, "F": 120, "G": 120, "H": 120,
                         "I": 35, "J": 80, "K": 100, "L": 80,
                         "M": 100, "N": 100, "O": 100, "P": 120,
                         "Q": 120, "R": 130, "S": 110, "T": 90,
                         "U": 100, "V": 100, "W": 100, "X": 130,
                         "Y": 120, "Z": 120,
                         "a": 120, "b": 120, "c": 100, "d": 100,
                         "e": 120, "f": 120, "g": 120, "h": 120,
                         "i": 35, "j": 80, "k": 100, "l": 80,
                         "m": 100, "n": 100, "o": 100, "p": 120,
                         "q": 120, "r": 130, "s": 110, "t": 90,
                         "u": 100, "v": 100, "w": 100, "x": 130,
                         "y": 120, "z": 120}
            target_cp = {k: v * level_of_cc for k, v in target_cp.items()}

        print("init_path: ", init_path)

        subdivision_thresh = None
        self.font_string_to_svgs(init_path,
                                 font_path,
                                 word,
                                 target_control=target_cp,
                                 subdivision_thresh=subdivision_thresh)
        self.normalize_letter_size(init_path, font_path, word)

        # optimize two adjacent letters
        print("letter: ", letter)
        if len(letter) > 1:
            subdivision_thresh = None
            self.font_string_to_svgs(init_path,
                                     font_path,
                                     letter,
                                     target_control=target_cp,
                                     subdivision_thresh=subdivision_thresh)
            self.normalize_letter_size(init_path, font_path, letter)

        print("preprocess_font done.")

    def font_string_to_svgs(self, dest_path, font, txt, size=30, spacing=1.0, target_control=None,
                            subdivision_thresh=None):
        fontname = self.font
        glyph_beziers = font_string_to_beziers(font, txt, size, spacing, merge=False, target_control=target_control)

        # compute bounding box
        points = np.vstack(sum(glyph_beziers, []))
        lt = np.min(points, axis=0)
        rb = np.max(points, axis=0)
        size = rb - lt

        sizestr = 'width="%.1f" height="%.1f"' % (size[0], size[1])
        boxstr = ' viewBox="%.1f %.1f %.1f %.1f"' % (lt[0], lt[1], size[0], size[1])
        header = '''<?xml version="1.0" encoding="utf-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" baseProfile="full" '''
        header += sizestr
        header += boxstr
        header += '>\n<defs/>\n'

        svg_all = header

        for i, (c, beziers) in enumerate(zip(txt, glyph_beziers)):
            fname, path = write_letter_svg(c, header, fontname, beziers, subdivision_thresh, dest_path)

            num_cp = self.count_cp(fname)
            print(f"Total control point: {num_cp} -- {c}")
            # Add to global svg
            svg_all += path + '</g>\n'

        # Save global svg
        svg_all += '</svg>\n'
        fname = f"{dest_path}/{fontname}_{txt}.svg"
        fname = fname.replace(" ", "_")
        with open(fname, 'w') as f:
            f.write(svg_all)

    def count_cp(self, file_name):
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(file_name)
        p_counter = 0
        for path in shapes:
            p_counter += path.points.shape[0]
        return p_counter

    def normalize_letter_size(self, dest_path, font, txt):
        fontname = os.path.splitext(os.path.basename(font))[0]
        for i, c in enumerate(txt):
            fname = f"{dest_path}/{fontname}_{c}.svg"
            fname = fname.replace(" ", "_")
            self.fix_single_svg(fname)

        fname = f"{dest_path}/{fontname}_{txt}.svg"
        fname = fname.replace(" ", "_")
        self.fix_single_svg(fname, all_word=True)

    def fix_single_svg(self, svg_path, all_word=False):
        target_h_letter = 360
        target_canvas_width, target_canvas_height = 600, 600

        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)

        letter_h = canvas_height
        letter_w = canvas_width

        if all_word:
            if letter_w > letter_h:
                scale_canvas_w = target_h_letter / letter_w
                hsize = int(letter_h * scale_canvas_w)
                scale_canvas_h = hsize / letter_h
            else:
                scale_canvas_h = target_h_letter / letter_h
                wsize = int(letter_w * scale_canvas_h)
                scale_canvas_w = wsize / letter_w
        else:
            scale_canvas_h = target_h_letter / letter_h
            wsize = int(letter_w * scale_canvas_h)
            scale_canvas_w = wsize / letter_w

        for num, p in enumerate(shapes):
            p.points[:, 0] = p.points[:, 0] * scale_canvas_w
            p.points[:, 1] = p.points[:, 1] * scale_canvas_h + target_h_letter

        w_min = min([torch.min(p.points[:, 0]) for p in shapes])
        w_max = max([torch.max(p.points[:, 0]) for p in shapes])
        h_min = min([torch.min(p.points[:, 1]) for p in shapes])
        h_max = max([torch.max(p.points[:, 1]) for p in shapes])

        for num, p in enumerate(shapes):
            p.points[:, 0] = p.points[:, 0] + (target_canvas_width / 2) - int(w_min + (w_max - w_min) / 2)
            p.points[:, 1] = p.points[:, 1] + (target_canvas_height / 2) - int(h_min + (h_max - h_min) / 2)

        output_path = f"{svg_path[:-4]}_scaled.svg"
        print("output_path: ", output_path)
        self.save_svg(output_path, target_canvas_width, target_canvas_height, shapes, shape_groups)

    def combine_word(self, word, letter, font, results_dir):
        word_svg_scaled = results_dir / f"{font}_{word}_scaled.svg"
        canvas_width_word, canvas_height_word, shapes_word, shape_groups_word = pydiffvg.svg_to_scene(word_svg_scaled)
        letter_ids = []
        for l in letter:
            letter_ids += self.get_letter_ids(l, word, shape_groups_word)

        w_min, w_max = min([torch.min(shapes_word[ids].points[:, 0]) for ids in letter_ids]), max(
            [torch.max(shapes_word[ids].points[:, 0]) for ids in letter_ids])
        h_min, h_max = min([torch.min(shapes_word[ids].points[:, 1]) for ids in letter_ids]), max(
            [torch.max(shapes_word[ids].points[:, 1]) for ids in letter_ids])

        c_w = (-w_min + w_max) / 2
        c_h = (-h_min + h_max) / 2

        svg_result = results_dir / "final_letter.svg"
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_result)

        out_w_min, out_w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max(
            [torch.max(p.points[:, 0]) for p in shapes])
        out_h_min, out_h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max(
            [torch.max(p.points[:, 1]) for p in shapes])

        out_c_w = (-out_w_min + out_w_max) / 2
        out_c_h = (-out_h_min + out_h_max) / 2

        scale_canvas_w = (w_max - w_min) / (out_w_max - out_w_min)
        scale_canvas_h = (h_max - h_min) / (out_h_max - out_h_min)

        if scale_canvas_h > scale_canvas_w:
            wsize = int((out_w_max - out_w_min) * scale_canvas_h)
            scale_canvas_w = wsize / (out_w_max - out_w_min)
            shift_w = -out_c_w * scale_canvas_w + c_w
        else:
            hsize = int((out_h_max - out_h_min) * scale_canvas_w)
            scale_canvas_h = hsize / (out_h_max - out_h_min)
            shift_h = -out_c_h * scale_canvas_h + c_h

        for num, p in enumerate(shapes):
            p.points[:, 0] = p.points[:, 0] * scale_canvas_w
            p.points[:, 1] = p.points[:, 1] * scale_canvas_h
            if scale_canvas_h > scale_canvas_w:
                p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min + shift_w
                p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min
            else:
                p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min
                p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min + shift_h

        for j, s in enumerate(letter_ids):
            shapes_word[s] = shapes[j]

        word_letter_result = results_dir / f"{font}_{word}_{letter}.svg"
        self.save_svg(word_letter_result, canvas_width, canvas_height, shapes_word, shape_groups_word)

        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width,
                                                             canvas_height,
                                                             shapes_word,
                                                             shape_groups_word)
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + \
              torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        word_letter_result = results_dir / f"{font}_{word}_{letter}.png"
        self.save_image(img, word_letter_result)

    def get_letter_ids(self, letter, word, shape_groups):
        for group, l in zip(shape_groups, word):
            if l == letter:
                return group.shape_ids

    def pretty_save_svg(self, filename, width=None, height=None, shapes=None, shape_groups=None):
        width = self.canvas_width if width is None else width
        height = self.canvas_height if height is None else height
        shapes = self.shapes if shapes is None else shapes
        shape_groups = self.shape_groups if shape_groups is None else shape_groups

        self.save_svg(filename, width, height, shapes, shape_groups, use_gamma=False, background=None)


class PainterOptimizer:

    def __init__(self, renderer, num_iter, lr_cfg):
        self.renderer = renderer
        self.num_iter = num_iter
        self.lr_cfg = lr_cfg
        self.lr_base = {'point': lr_cfg.point_lr}

        point_vars = self.renderer.get_point_parameters()
        self.para = {'point': point_vars}

        self.optimizer = None
        self.scheduler = None

    def init_optimizers(self):
        # optimizer
        learnable_params = [
            {'params': self.para[ki], 'lr': self.lr_base[ki]} for ki in sorted(self.para.keys())
        ]
        self.optimizer = torch.optim.Adam(learnable_params, betas=(0.9, 0.9), eps=1e-6)

        # lr schedule
        lr_lambda_fn = lambda step: learning_rate_decay(
            step,
            self.lr_cfg.lr_init,
            self.lr_cfg.lr_final,
            self.num_iter,
            self.lr_cfg.lr_delay_steps,
            self.lr_cfg.lr_delay_mult
        ) / self.lr_cfg.lr_init
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda_fn, last_epoch=-1)

    def update_lr(self):
        self.scheduler.step()

    def zero_grad_(self):
        self.optimizer.zero_grad()

    def step_(self):
        self.optimizer.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
    """
    Continuous learning rate decay function.
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    pytorch adaptation of https://github.com/google/mipnerf

    Args:
        step: int, the current optimization step.
        lr_init: float, the initial learning rate.
        lr_final: float, the final learning rate.
        max_steps: int, the number of steps during optimization.
        lr_delay_steps: int, the number of steps to delay the full learning rate.
        lr_delay_mult: float, the multiplier on the rate when delaying it.
    Returns:
        lr: the learning for current step 'step'.
    """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp
