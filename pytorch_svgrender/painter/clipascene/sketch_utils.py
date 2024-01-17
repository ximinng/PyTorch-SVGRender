from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pydiffvg
import torch
from PIL import Image
from pytorch_svgrender.diffvg_warp import DiffVGState
from pytorch_svgrender.painter.clipasso.u2net import U2NET
from skimage.transform import resize
from torchvision import transforms
from torchvision.utils import make_grid

from .painter_params import MLP, WidthMLP


def plot_attn_dino(attn, threshold_map, inputs, inds, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(2, attn.shape[0] + 2, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, 2)
    plt.imshow(attn.sum(0).numpy(), interpolation='nearest')
    plt.title("atn map sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 3)
    plt.imshow(threshold_map[-1].numpy(), interpolation='nearest')
    plt.title("prob sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 4)
    plt.imshow(threshold_map[:-1].sum(0).numpy(), interpolation='nearest')
    plt.title("thresh sum")
    plt.axis("off")

    for i in range(attn.shape[0]):
        plt.subplot(2, attn.shape[0] + 2, i + 3)
        plt.imshow(attn[i].numpy())
        plt.axis("off")
        plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 1 + i + 4)
        plt.imshow(threshold_map[i].numpy())
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_attn_clip(attn, threshold_map, inputs, inds, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("attn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / \
                     (threshold_map.max() - threshold_map.min())
    plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_attn(attn, threshold_map, inputs, inds, output_path, saliency_model):
    if saliency_model == "dino":
        plot_attn_dino(attn, threshold_map, inputs, inds, output_path)
    elif saliency_model == "clip":
        plot_attn_clip(attn, threshold_map, inputs, inds, output_path)


def fix_image_scale(im):
    im_np = np.array(im) / 255
    height, width = im_np.shape[0], im_np.shape[1]
    max_len = max(height, width) + 20
    new_background = np.ones((max_len, max_len, 3))
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2
    new_background[y: y + height, x: x + width] = im_np
    new_background = (new_background / new_background.max() * 255).astype(np.uint8)
    new_im = Image.fromarray(new_background)
    return new_im


def get_mask_u2net(pil_im, output_dir, u2net_path, device="cpu"):
    # input preprocess
    w, h = pil_im.size[0], pil_im.size[1]
    im_size = min(w, h)
    data_transforms = transforms.Compose([
        transforms.Resize(min(320, im_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    input_im_trans = data_transforms(pil_im).unsqueeze(0).to(device)

    # load U^2 Net model
    net = U2NET(in_ch=3, out_ch=1)
    net.load_state_dict(torch.load(u2net_path))
    net.to(device)
    net.eval()

    # get mask
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(input_im_trans.detach())
    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1
    mask = torch.cat([predict, predict, predict], dim=0).permute(1, 2, 0)
    mask = mask.cpu().numpy()
    mask = resize(mask, (h, w), anti_aliasing=False)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    # predict_np = predict.clone().cpu().data.numpy()
    im = Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8)).convert('RGB')
    save_path_ = output_dir / "mask.png"
    im.save(save_path_)

    im_np = np.array(pil_im)
    im_np = im_np / im_np.max()
    im_np = mask * im_np
    im_np[mask == 0] = 1
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)

    # free u2net
    del net
    torch.cuda.empty_cache()

    return im_final, predict


def log_best_normalised_sketch(configs_to_save, output_dir, eval_interval, min_eval_iter):
    np.save(f"{output_dir}/config.npy", configs_to_save)
    losses_eval = {}
    for k in configs_to_save.keys():
        if "_original_eval" in k and "normalization" not in k:
            cur_arr = np.array(configs_to_save[k])
            mu = cur_arr.mean()
            std = cur_arr.std()
            losses_eval[k] = (cur_arr - mu) / std

    final_normalise_losses = sum(list(losses_eval.values()))
    sorted_iters = np.argsort(final_normalise_losses)
    index = 0
    best_iter = sorted_iters[index]
    best_normalised_loss = final_normalise_losses[best_iter]
    best_num_strokes = configs_to_save["num_strokes"][best_iter]

    iter_ = best_iter * eval_interval + min_eval_iter
    configs_to_save["best_normalised_iter"] = iter_
    configs_to_save["best_normalised_loss"] = best_normalised_loss
    configs_to_save["best_normalised_num_strokes"] = best_num_strokes
    copyfile(output_dir / "mlps" / f"points_mlp{iter_}.pt", output_dir / "points_mlp.pt")
    copyfile(output_dir / "mlps" / f"width_mlp{iter_}.pt", output_dir / "width_mlp.pt")


def inference_sketch(args, output_dir, eps=1e-4, device="cpu"):
    mlp_points_weights_path = output_dir / "points_mlp.pt"
    mlp_width_weights_path = output_dir / "width_mlp.pt"
    sketch_init_path = output_dir / "svg_logs" / "init_svg.svg"
    output_path = output_dir

    num_paths = args.num_paths
    control_points_per_seg = args.control_points_per_seg
    width_ = 1.5
    num_control_points = torch.zeros(1, dtype=torch.int32) + (control_points_per_seg - 2)
    init_widths = torch.ones((num_paths)).to(device) * width_

    mlp = MLP(num_strokes=num_paths, num_cp=control_points_per_seg).to(device)
    checkpoint = torch.load(mlp_points_weights_path)
    mlp.load_state_dict(checkpoint['model_state_dict'])

    if args.width_optim:
        mlp_width = WidthMLP(num_strokes=num_paths, num_cp=control_points_per_seg).to(device)
        checkpoint = torch.load(mlp_width_weights_path)
        mlp_width.load_state_dict(checkpoint['model_state_dict'])

    points_vars, canvas_width, canvas_height = get_init_points(sketch_init_path)
    points_vars = torch.stack(points_vars).unsqueeze(0).to(device)
    points_vars = points_vars / canvas_width
    points_vars = 2 * points_vars - 1
    points = mlp(points_vars)

    all_points = 0.5 * (points + 1.0) * canvas_width
    all_points = all_points + eps * torch.randn_like(all_points)
    all_points = all_points.reshape((-1, num_paths, control_points_per_seg, 2))

    if args.width_optim:  # first iter use just the location mlp
        widths_ = mlp_width(init_widths).clamp(min=1e-8)
        mask_flipped = (1 - widths_).clamp(min=1e-8)
        v = torch.stack((torch.log(widths_), torch.log(mask_flipped)), dim=-1)
        hard_mask = torch.nn.functional.gumbel_softmax(v, 0.2, False)
        stroke_probs = hard_mask[:, 0]
        widths = stroke_probs * init_widths

    shapes = []
    shape_groups = []
    for p in range(num_paths):
        width = torch.tensor(width_)
        if args.width_optim:
            width = widths[p]
        w = width / 1.5
        path = pydiffvg.Path(
            num_control_points=num_control_points, points=all_points[:, p].reshape((-1, 2)),
            stroke_width=width, is_closed=False)
        is_in_canvas_ = is_in_canvas(canvas_width, canvas_height, path, device)
        if is_in_canvas_ and w > 0.7:
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=torch.tensor([0, 0, 0, 1]))
            shape_groups.append(path_group)
    pydiffvg.save_svg(output_path / "best_iter.svg", canvas_width, canvas_height, shapes, shape_groups)


def get_init_points(path_svg):
    points_init = []
    canvas_width, canvas_height, shapes, shape_groups = DiffVGState.load_svg(path_svg)
    for path in shapes:
        points_init.append(path.points)
    return points_init, canvas_width, canvas_height


def is_in_canvas(canvas_width, canvas_height, path, device):
    shapes, shape_groups = [], []
    stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
    shapes.append(path)
    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                     fill_color=None,
                                     stroke_color=stroke_color)
    shape_groups.append(path_group)
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
    img = img[:, :, :3].detach().cpu().numpy()
    return (1 - img).sum()
