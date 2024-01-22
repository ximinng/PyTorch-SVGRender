import matplotlib.pyplot as plt
import numpy as np
import pydiffvg
import torch
from PIL import Image
from pytorch_svgrender.painter.clipascene import u2net_utils
from pytorch_svgrender.painter.clipasso.u2net import U2NET
from scipy import ndimage
from skimage import morphology
from skimage.measure import label
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid


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


def get_size_of_largest_cc(binary_im):
    labels, num = label(binary_im, background=0, return_num=True)
    (unique, counts) = np.unique(labels, return_counts=True)
    args = np.argsort(counts)[::-1]
    largest_cc_label = unique[args][1]  # without background
    return counts[args][1]


def get_num_cc(binary_im):
    labels, num = label(binary_im, background=0, return_num=True)
    return num


def get_obj_bb(binary_im):
    y = np.where(binary_im != 0)[0]
    x = np.where(binary_im != 0)[1]
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    return x0, x1, y0, y1


def cut_and_resize(im, x0, x1, y0, y1, new_height, new_width):
    cut_obj = im[y0: y1, x0: x1]
    resized_obj = resize(cut_obj, (new_height, new_width))
    new_mask = np.zeros(im.shape)
    center_y_new = int(new_height / 2)
    center_x_new = int(new_width / 2)
    center_targ_y = int(new_mask.shape[0] / 2)
    center_targ_x = int(new_mask.shape[1] / 2)
    startx, starty = center_targ_x - center_x_new, center_targ_y - center_y_new
    new_mask[starty: starty + resized_obj.shape[0], startx: startx + resized_obj.shape[1]] = resized_obj
    return new_mask


def get_mask_u2net(pil_im, output_dir, u2net_path, resize_obj=0, preprocess=False, device="cpu"):
    w, h = pil_im.size[0], pil_im.size[1]

    test_salobj_dataset = u2net_utils.SalObjDataset(imgs_list=[pil_im],
                                                    lbl_name_list=[],
                                                    transform=transforms.Compose([u2net_utils.RescaleT(320),
                                                                                  u2net_utils.ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    input_im_trans = next(iter(test_salobj_dataloader))

    net = U2NET(3, 1)
    net.load_state_dict(torch.load(u2net_path))
    net.to(device)
    net.eval()

    with torch.no_grad():
        input_im_trans = input_im_trans.type(torch.FloatTensor)
        d1, d2, d3, d4, d5, d6, d7 = net(input_im_trans.cuda())

    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred

    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1

    if preprocess:
        predict = torch.tensor(
            ndimage.binary_dilation(predict[0].cpu().numpy(), structure=np.ones((11, 11))).astype(int)).unsqueeze(0)

        mask = torch.cat([predict, predict, predict], axis=0).permute(1, 2, 0)
        mask = mask.cpu().numpy()
        max_val = mask.max()
        mask[mask > max_val / 2] = 255
        mask = mask.astype(np.uint8)
        mask = resize(mask, (h, w), anti_aliasing=False, order=0)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        return mask

    mask = torch.cat([predict, predict, predict], axis=0).permute(1, 2, 0)
    mask = mask.cpu().numpy()
    mask = resize(mask, (h, w), anti_aliasing=False)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    im = Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8)).convert('RGB')
    im.save(output_dir / "mask.png")
    im_np = np.array(pil_im)
    im_np = im_np / im_np.max()

    if resize_obj:
        params = {}
        mask_np = mask[:, :, 0].astype(int)
        target_np = im_np
        min_size = int(get_size_of_largest_cc(mask_np) / 3)
        mask_np2 = morphology.remove_small_objects((mask_np > 0), min_size=min_size).astype(int)
        num_cc = get_num_cc(mask_np2)

        mask_np3 = np.ones((h, w, 3))
        mask_np3[:, :, 0] = mask_np2
        mask_np3[:, :, 1] = mask_np2
        mask_np3[:, :, 2] = mask_np2

        x0, x1, y0, y1 = get_obj_bb(mask_np2)

        im_width, im_height = x1 - x0, y1 - y0
        max_size = max(im_width, im_height)
        target_size = int(min(h, w) * 0.7)

        if max_size < target_size and num_cc == 1:
            if im_width > im_height:
                new_width, new_height = target_size, int((target_size / im_width) * im_height)
            else:
                new_width, new_height = int((target_size / im_height) * im_width), target_size
            mask = cut_and_resize(mask_np3, x0, x1, y0, y1, new_height, new_width)
            target_np = target_np / target_np.max()
            im_np = cut_and_resize(target_np, x0, x1, y0, y1, new_height, new_width)

            params["original_center_y"] = (y0 + (y1 - y0) / 2) / h
            params["original_center_x"] = (x0 + (x1 - x0) / 2) / w
            params["scale_w"] = new_width / im_width
            params["scale_h"] = new_height / im_height

        np.save(output_dir / "resize_params.npy", params)

    im_np = mask * im_np
    im_np[mask == 0] = 1
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)

    return im_final, mask


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
