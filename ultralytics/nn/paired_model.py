import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

class PairedDetectionModel(DetectionModel):
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        self._cached_features = None
        self.model[-1].register_forward_pre_hook(self._cache_head_input)

    def init_criterion(self):
        return PairedLoss(self)

    def extract_feature(self, x):
        y = []
        for m in self.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)

        head = self.model[-1]
        features = [y[j] for j in head.f]
        return features

    def _cache_head_input(self, module, inputs):
        if self.training:
            self._cached_features = list(inputs[0])

class PairedLoss(v8DetectionLoss):
    def __call__(self, preds, batch):
        loss, loss_item = super().__call__(preds, batch)
        if "gt_img" in batch:
            f_list = self.model._cached_features
            self.model._cached_features = None

            with torch.no_grad():
                f_gt_list = self.model.extract_feature(batch["gt_img"])

            dist_loss = torch.tensor(0.0, device=self.device)
            mse = nn.MSELoss()
            for f, f_gt in zip(f_list, f_gt_list):
                dist_loss += mse(f, f_gt)
 
            loss += 0.1 * dist_loss
        return loss, loss_item