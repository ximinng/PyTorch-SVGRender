# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

from typing import Union, List
from pathlib import Path
from datetime import datetime
import logging

from omegaconf import OmegaConf, DictConfig
from pprint import pprint
import torch
from accelerate.utils import LoggerType
from accelerate import Accelerator

from ..utils.logging import get_logger


class ModelState:
    """
    Handling logger and `hugging face` accelerate training

    features:
        - Precision
        - Device
        - Optimizer
        - Logger (default: python system print and logging)
        - Monitor (default: wandb, tensorboard)
    """

    def __init__(
            self,
            args: DictConfig,
            log_path_suffix: str = None,
            ignore_log=False,  # whether to create log file or not
    ) -> None:
        self.args: DictConfig = args
        # set cfg
        self.state_cfg = args.state
        self.x_cfg = args.x

        """check valid"""
        mixed_precision = self.state_cfg.get("mprec")
        # Bug: omegaconf convert 'no' to false
        mixed_precision = "no" if type(mixed_precision) == bool else mixed_precision

        """create working space"""
        # rule: ['./config'. 'method_name', 'exp_name.yaml']
        # -> result_path: ./runs/{method_name}-{exp_name}, as a base folder
        now_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
        results_folder = self.args.get("result_path", None)
        if results_folder is None:
            self.result_path = Path("./workdir") / f"{self.x_cfg.method}-{now_time}"
        else:
            self.result_path = Path(results_folder) / f"{self.x_cfg.method}-{now_time}"

        # update result_path: ./runs/{method_name}-{exp_name}/{log_path_suffix}
        # noting: can be understood as "results dir / methods / ablation study / your result"
        config_name_only = str(self.x_cfg.method).split(".")[0]
        if log_path_suffix is not None:
            self.result_path = self.result_path / f"{config_name_only}-{log_path_suffix}"
        else:
            self.result_path = self.result_path / f"{config_name_only}"

        """init visualized tracker"""
        # TODO: monitor with WANDB or TENSORBOARD
        self.log_with = []
        # if self.state_cfg.wandb:
        #     self.log_with.append(LoggerType.WANDB)
        # if self.state_cfg.tensorboard:
        #     self.log_with.append(LoggerType.TENSORBOARD)

        """HuggingFace Accelerator"""
        self.accelerator = Accelerator(
            device_placement=True,
            mixed_precision=mixed_precision,
            cpu=True if self.state_cfg.cpu else False,
            log_with=None if len(self.log_with) == 0 else self.log_with,
            project_dir=self.result_path / "vis",
        )

        """logs"""
        if self.accelerator.is_local_main_process:
            # logging
            self.log = logging.getLogger(__name__)

            # log results in a folder periodically
            self.result_path.mkdir(parents=True, exist_ok=True)
            if not ignore_log:
                self.logger = get_logger(
                    logs_dir=self.result_path.as_posix(),
                    file_name=f"{now_time}-{args.seed}-log.txt"
                )

            print("==> system args: ")
            sys_cfg = OmegaConf.masked_copy(args, ["x"])
            print(sys_cfg)
            print("==> yaml config args: ")
            print(self.x_cfg)

            print("\n***** Model State *****")
            print(f"-> Mixed Precision: {mixed_precision}, AMP: {self.accelerator.native_amp}")
            print(f"-> Weight dtype:  {self.weight_dtype}")

            if self.accelerator.scaler_handler is not None and self.accelerator.scaler_handler.enabled:
                print(f"-> Enabled GradScaler: {self.accelerator.scaler_handler.to_kwargs()}")

            print(f"-> Working Space: '{self.result_path}'")

        """glob step"""
        self.step = 0

        """log process"""
        self.accelerator.wait_for_everyone()
        print(f'Process {self.accelerator.process_index} using device: {self.accelerator.device}')

        self.print("-> state initialization complete \n")

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    @property
    def weight_dtype(self):
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        return weight_dtype

    @property
    def n_gpus(self):
        return self.accelerator.num_processes

    @property
    def no_decay_params_names(self):
        no_decay = [
            "bn", "LayerNorm", "GroupNorm",
        ]
        return no_decay

    def no_decay_params(self, model, weight_decay):
        """optimization tricks"""
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in self.no_decay_params_names)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in self.no_decay_params_names)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def optimized_params(self, model: torch.nn.Module, verbose=True) -> List:
        """return parameters if `requires_grad` is True

        Args:
            model: pytorch models
            verbose: log optimized parameters

        Examples:
            >>> params_optimized = self.optimized_params(uvit, verbose=True)
            >>> optimizer = torch.optim.AdamW(params_optimized, lr=1e-3)

        Returns:
                a list of parameters
        """
        params_optimized = []
        for key, value in model.named_parameters():
            if value.requires_grad:
                params_optimized.append(value)
                if verbose:
                    self.print("\t {}, {}, {}".format(key, value.numel(), value.shape))
        return params_optimized

    def save_everything(self, fpath: str):
        """Saving and loading the model, optimizer, RNG generators, and the GradScaler."""
        if not self.accelerator.is_main_process:
            return
        self.accelerator.save_state(fpath)

    def load_save_everything(self, fpath: str):
        """Loading the model, optimizer, RNG generators, and the GradScaler."""
        self.accelerator.load_state(fpath)

    def save(self, milestone: Union[str, float, int], checkpoint: object) -> None:
        if not self.accelerator.is_main_process:
            return

        torch.save(checkpoint, self.result_path / f'model-{milestone}.pt')

    def save_in(self, root: Union[str, Path], checkpoint: object) -> None:
        if not self.accelerator.is_main_process:
            return

        torch.save(checkpoint, root)

    def load_ckpt_model_only(self, model: torch.nn.Module, path: Union[str, Path], rm_module_prefix: bool = False):
        ckpt = torch.load(path, map_location=self.device)

        unwrapped_model = self.accelerator.unwrap_model(model)
        if rm_module_prefix:
            unwrapped_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
        else:
            unwrapped_model.load_state_dict(ckpt)
        return unwrapped_model

    def load_shared_weights(self, model: torch.nn.Module, path: Union[str, Path]):
        ckpt = torch.load(path, map_location=self.accelerator.device)
        self.print(f"pretrained_dict len: {len(ckpt)}")
        unwrapped_model = self.accelerator.unwrap_model(model)
        model_dict = unwrapped_model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        unwrapped_model.load_state_dict(model_dict, strict=False)
        self.print(f"selected pretrained_dict: {len(model_dict)}")
        return unwrapped_model

    def print(self, *args, **kwargs):
        """Use in replacement of `print()` to only print once per server."""
        self.accelerator.print(*args, **kwargs)

    def pretty_print(self, msg):
        if self.accelerator.is_main_process:
            pprint(dict(msg))

    def close_tracker(self):
        self.accelerator.end_training()

    def free_memory(self):
        self.accelerator.clear()

    def close(self, msg: str = "Training complete."):
        """Use in end of training."""
        self.free_memory()

        if torch.cuda.is_available():
            self.print(f'\nGPU memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
        if len(self.log_with) > 0:
            self.close_tracker()
        self.print(msg)
