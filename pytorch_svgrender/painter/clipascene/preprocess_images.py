# ===================================================
# ================= preprocess ======================
# ===================================================
# This script preprocess the input images.
# Spesifically, we divie the image into two regions - foreground and background using U2Net
# if you want to provide your own mask, use this script with --run_u2net 0
# Otherwise, we will automatically generate the mask for you.
# Then we use LAMA inpainting to fill the missing areas for the background image.
# Example of a running command:
# CUDA_VISIBLE_DEVICES=6 python preprocess_images.py --im_name "man_flowers.png"
# ===================================================
import argparse
import logging
import os
import shutil
import sys
import traceback

import imageio
from PIL import Image
from lama.saicinpainting.evaluation.refinement import refine_predict
from lama.saicinpainting.evaluation.utils import move_to_device
from scipy import ndimage
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision import transforms

import u2net_utils
from pytorch_svgrender.painter.clipasso.u2net import U2NET

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from lama.saicinpainting.training.data.datasets import make_default_val_dataset
from lama.saicinpainting.training.trainers import load_checkpoint

LOGGER = logging.getLogger(__name__)


def get_U2Net_mask(top_path, im_name, device, use_gpu):
    im = Image.open(f"{top_path}/{im_name}")
    w, h = im.size[0], im.size[1]

    test_salobj_dataset = u2net_utils.SalObjDataset(imgs_list=[im],
                                                    lbl_name_list=[],
                                                    transform=transforms.Compose([u2net_utils.RescaleT(320),
                                                                                  u2net_utils.ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    input_im_trans = next(iter(test_salobj_dataloader))

    model_dir = os.path.join("U2Net_/saved_models/u2net.pth")
    net = U2NET(3, 1)
    if torch.cuda.is_available() and use_gpu:
        net.load_state_dict(torch.load(model_dir))
        net.to(device)
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    with torch.no_grad():
        input_im_trans = input_im_trans.type(torch.FloatTensor)
        d1, d2, d3, d4, d5, d6, d7 = net(input_im_trans.to(device))

    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred

    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1

    predict = torch.tensor(
        ndimage.binary_dilation(predict[0].cpu().numpy(), structure=np.ones((11, 11))).astype(np.int)).unsqueeze(0)

    mask = torch.cat([predict, predict, predict], axis=0).permute(1, 2, 0)
    mask = mask.cpu().numpy()
    max_val = mask.max()
    mask[mask > max_val / 2] = 255
    mask = mask.astype(np.uint8)
    mask = resize(mask, (h, w), anti_aliasing=False, order=0)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    return mask


def apply_inpaint(predict_config, device):
    try:
        # register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path,
                                       'models',
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            print(mask_fname)
            cur_out_fname = os.path.join(
                predict_config.outdir,
                os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = model(batch)
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


def standarlized_images(top_path, im_name):
    if os.path.splitext(im_name)[1] in [".png", ".jpg", ".jpeg"]:
        im = np.array(Image.open(f"{top_path}/{im_name}"))
        h, w = im.shape[0], im.shape[1]
        if h != w:
            print(f"!! Note - {im_name} size is {h}x{w}, the image is not square, image will be resized !!")
        max_size = max(h, w)
        if max_size > 512:
            print(f"!! Note - {im_name} size is {h}x{w}, which is too large, image will be resized to 512x512 !!")
            im = Image.open(f"{top_path}/{im_name}").convert("RGB").resize((512, 512))
            im.save(f"{top_path}/{os.path.splitext(im_name)[0]}.png")
        # replace to png for LAMA
        elif os.path.splitext(im_name)[1] in [".jpg", ".jpeg"]:
            print(f"!! Note - {im_name} is not png, will be replaced to png !!")
            input_path = f"{args.top_path}/{im_name}"
            copy_path = f"{args.top_path}/{os.path.splitext(im_name)[0]}.png"
            shutil.copyfile(input_path, copy_path)


if __name__ == "__main__":
    # TODO - aplly this on the entire folder without image name
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_u2net", type=int, default=1)
    parser.add_argument("--top_path", type=str, default="./target_images/scene")
    parser.add_argument("--use_gpu", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

    for im_name in os.listdir(args.top_path):
        standarlized_images(args.top_path, im_name)

    if args.run_u2net:
        print("Producing mask using U2Net....")
        for im_name in os.listdir(args.top_path):
            if os.path.splitext(im_name)[1] in [".png"] and "_mask" not in os.path.splitext(im_name)[0]:
                output_path = f"{args.top_path}/{os.path.splitext(im_name)[0]}_mask.png"
                if os.path.exists(output_path):
                    print(f"{output_path} already exists!")
                else:
                    mask = get_U2Net_mask(args.top_path, im_name, device, args.use_gpu)
                    imageio.imsave(output_path, mask)
                    print(f"Mask generated successfully! and saved to {output_path}")

    # running LAMA
    print("=" * 50)
    print("Applying LAMA inpainting....")
    conf = OmegaConf.load('lama/configs/prediction/default.yaml')
    conf.model.path = "lama/big-lama"
    conf.indir = args.top_path
    conf.outdir = "./target_images/background/"
    apply_inpaint(conf, device)
