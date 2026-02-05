import math
import os
import random
from copy import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataloader, distributed

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.paired_model import PairedDetectionModel
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK

class PairedDectectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build Paired YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        gt_img_path = self.data.get("train_gt") if mode == "train" else None
        return build_yolo_dataset(self.args, img_path, batch, self.data, gt_img_path=gt_img_path, mode=mode, rect=mode == "val", stride=gs, dehazing=True)
    
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch = super().preprocess_batch(batch)
        # Custom processing for 'gt_img' if it exists
        if "gt_img" in batch:
            batch["gt_img"] = batch["gt_img"].to(self.device, non_blocking=True).float() / 255
            # WARNING: If multi_scale was applied to 'img' in super(), 'gt_img' is now mismatched size.
            # You should probably disable multi_scale in args for now.
        
        return batch


    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = PairedDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model