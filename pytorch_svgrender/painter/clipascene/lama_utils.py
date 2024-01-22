from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
import yaml
from lama.saicinpainting.evaluation.refinement import refine_predict
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.data.datasets import make_default_val_dataset
from lama.saicinpainting.training.trainers import load_checkpoint
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate


def apply_inpaint(scene_path, background_path, device):
    conf = OmegaConf.load('lama/configs/prediction/default.yaml')
    model_path = Path("lama/big-lama")
    train_config_path = model_path / 'config.yaml'
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    out_ext = conf.get('out_ext', '.png')

    checkpoint_path = model_path / 'models' / conf.model.checkpoint
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not conf.get('refine', False):
        model.to(device)

    dataset = make_default_val_dataset(scene_path, **conf.dataset)
    for img_i in tqdm.trange(len(dataset)):
        mask_fname = Path(dataset.mask_filenames[img_i])
        relative_fname = mask_fname.relative_to(scene_path).with_suffix(out_ext)
        cur_out_fname = background_path / relative_fname
        cur_out_fname.parent.mkdir(parents=True, exist_ok=True)
        batch = default_collate([dataset[img_i]])
        if conf.get('refine', False):
            assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
            # image unpadding is taken care of in the refiner, so that output image
            # is same size as the input image
            cur_res = refine_predict(batch, model, **conf.refiner)
            cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
        else:
            with torch.no_grad():
                batch = move_to_device(batch, device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model(batch)
                cur_res = batch[conf.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cur_out_fname.as_posix(), cur_res)
