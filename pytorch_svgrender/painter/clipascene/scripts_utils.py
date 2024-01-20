import os

import numpy as np
import pydiffvg
import torch
from scipy.optimize import curve_fit


def get_svg_file(path):
    files = os.listdir(path)
    files = [f for f in files if "best.svg" in f]
    return files[0]


def get_seed(filename):
    filename = filename[:-9]
    keyword = 'seed'
    before_keyword, keyword, after_keyword = filename.partition(keyword)
    return after_keyword


def get_clip_loss(path, layer):
    path_config = path / "config.npy"
    config = np.load(path_config, allow_pickle=True)[()]
    loss_clip = np.array(config[f"loss_eval"])
    best_iter = np.argsort(loss_clip)[0]
    loss_clip_layer = np.array(config[f"clip_vit_l{layer}_original_eval"])
    return loss_clip, best_iter, loss_clip_layer


def ratios_to_str(ratios):
    ratios_str = ""
    for r_ in ratios:
        r_str = f"{r_:.3f}"
        ratios_str += f"{float(r_str)},"
    ratios_str = ratios_str[:-1]
    return ratios_str


def func(x, a, c, d):
    return a * np.exp(c * x)


def func_inv(y, a, c, d):
    return np.log(y / a) * (1 / c)


def get_func(ratios_rel, start_x, start_ys):
    target_ys = ratios_rel[start_ys:]
    x = np.linspace(start_x, start_x + len(target_ys) - 1, len(target_ys))
    # calculate exponent
    popt, pcov = curve_fit(func, x, target_ys, maxfev=3000)
    return popt


def get_clip_loss2(path, layer, object_or_background):
    path_config = path / "config.npy"
    config = np.load(path_config, allow_pickle=True)[()]
    loss_clip = np.array(config[f"loss_eval"])
    best_iter = np.argsort(loss_clip)[0]
    loss_clip_layer = np.array(config[f"clip_vit_l{layer}_original_eval"])
    if object_or_background == "object":
        loss_clip_layer4 = np.array(config[f"clip_vit_l4_original_eval"])
        loss_clip_layer = 1 * loss_clip_layer4 + loss_clip_layer
    return best_iter, loss_clip_layer


def get_ratios_dict(path_to_initial_sketches, folder_name_l, layer, im_name, object_or_background, step_size_l,
                    num_ratios=8):
    # get the sketch of the given layer, and get L_clip_i
    svg_filename = get_svg_file(path_to_initial_sketches / folder_name_l)
    seed = get_seed(svg_filename)
    path_li = path_to_initial_sketches / folder_name_l / f"{folder_name_l}_seed{seed}"
    best_iter, loss_clip_layer = get_clip_loss2(path_li, layer, object_or_background)
    best_lclip_layer = loss_clip_layer[best_iter]
    r_1_k = 1 / best_lclip_layer

    # get the next ratios by jumping by 2
    r_j_k = r_1_k
    ratios_k = [r_1_k]
    for j in range(4):
        r_j_k = r_j_k / 2
        ratios_k.append(r_j_k)
    start_ys, start_x, end_x_addition = 0, 0, 0
    popt = get_func(ratios_k, start_x=0, start_ys=0)  # fit the function to ratios_k
    x_1_k = func_inv([r_1_k], *popt)

    step_size = step_size_l
    num_steps = num_ratios - start_x + end_x_addition
    start_ = x_1_k[0]
    end = num_steps * step_size
    # sample the function from the initial scaled r_1 with the corresponding step size
    new_xs_layer_l = np.linspace(start_, end - step_size + start_, num_steps)
    # print("new_xs_layer_l", new_xs_layer_l)
    ratios_li = func(new_xs_layer_l, *popt)
    ratios_str = ratios_to_str(ratios_li)
    xs_layer_l_str = ratios_to_str(new_xs_layer_l)
    print(f"layer {layer} r_1_k {r_1_k} \n new {ratios_str} \n x {xs_layer_l_str}\n")
    return ratios_str


def read_svg(path_svg, multiply=0, resize_obj=False, params=None, opacity=1, device=None):
    pydiffvg.set_device(device)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(path_svg)
    for group in shape_groups:
        group.stroke_color = torch.tensor([0, 0, 0, opacity])
    if resize_obj and params:
        w, h = params["scale_w"], params["scale_h"]
        for path in shapes:
            path.points = path.points / canvas_width
            path.points = 2 * path.points - 1
            path.points[:, 0] /= (w)  # / canvas_width)
            path.points[:, 1] /= (h)  # / canvas_height)
            path.points = 0.5 * (path.points + 1.0) * canvas_width
            center_x, center_y = canvas_width / 2, canvas_height / 2
            path.points[:, 0] += (params["original_center_x"] * canvas_width - center_x)
            path.points[:, 1] += (params["original_center_y"] * canvas_height - center_y)
    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= multiply
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,  # num_samples_x
                  2,  # num_samples_y
                  0,  # seed
                  None,
                  *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
          torch.ones(img.shape[0], img.shape[1], 3,
                     device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3].cpu().numpy()
    return img
