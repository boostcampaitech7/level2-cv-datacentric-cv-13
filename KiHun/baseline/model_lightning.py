import torch
import numpy as np

import pytorch_lightning as pl

from torch import cuda
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

import wandb

from model import EAST

from deteval import calc_deteval_metrics
from detect import get_bboxes

class EASTLightningModel(pl.LightningModule):
    def __init__(self, image_size=2048, input_size=1024, learning_rate=1e-3, max_epoch=150, nowandb=False):
        super().__init__()
        self.model = EAST()
        self.image_size = image_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.nowandb = nowandb

        self.epoch_loss = []

        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []

    def _update_validation_metrics(self, metrics):
        self.val_precisions.append(metrics['precision'])
        self.val_recalls.append(metrics['recall'])
        self.val_f1s.append(metrics['hmean'])

    def _get_bboxes_dict(self, score_maps, geo_maps, orig_sizes, input_size):

        score_maps = score_maps.cpu().numpy()
        geo_maps = geo_maps.cpu().numpy()

        by_sample_bboxes = []
        for score_map, geo_map, orig_size in zip(score_maps, geo_maps, orig_sizes):

            bboxes = get_bboxes(score_map, geo_map)
            if bboxes is None:
                bboxes = np.zeros((0, 4, 2), dtype=np.float32)
            else:
                bboxes = bboxes[:, :8].reshape(-1, 4, 2)
                bboxes *= max(orig_size) / input_size

            by_sample_bboxes.append(bboxes)
    
        return dict(enumerate(by_sample_bboxes))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, gt_score_map, gt_geo_map, roi_mask = batch
        loss, extra_info = self.model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
        val_dict = {
            'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
            'IoU loss': extra_info['iou_loss']
        }
        self.epoch_loss.append(loss.item())
        self.log('train_loss', loss)
        if not self.nowandb:
            wandb.log(val_dict)
        return loss

    def on_train_epoch_end(self):
        mean_loss = np.array(self.epoch_loss).mean()
        if not self.nowandb:
            wandb.log({"Mean loss": mean_loss})

        self.epoch_loss.clear()

    def validation_step(self, batch, batch_idx):
        img, gt_score_map, gt_geo_map, roi_mask = batch
        _, extra_info = self.model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

        pred_score_map = extra_info["score_map"]
        pred_geo_map = extra_info["geo_map"]

        orig_sizes = [image.shape[-3:] for image in img]

        gt_bboxes_dict = self._get_bboxes_dict(gt_score_map, gt_geo_map, orig_sizes, self.input_size)
        pred_bboxes_dict = self._get_bboxes_dict(pred_score_map, pred_geo_map, orig_sizes, self.input_size)

        metrics = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)['total']
        self._update_validation_metrics(metrics)

        # 결과 반환 (손실 없이 metric에만 집중)
        return {'val_precision': metrics['precision'], 'val_recall': metrics['recall'], 'val_f1': metrics['hmean']}

    def on_validation_epoch_end(self):
        # 모든 batch에서 얻은 precision, recall, f1 metric의 평균 계산
        precision = np.array(self.val_precisions).mean()
        recall = np.array(self.val_recalls).mean()
        f1 = np.array(self.val_f1s).mean()

        # wandb와 로그에 기록
        if not self.nowandb:
            wandb.log({"val_precision": precision, "val_recall": recall, "val_f1": f1})

        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

        self.val_precisions.clear()
        self.val_recalls.clear()
        self.val_f1s.clear()

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = MultiStepLR(optimizer, milestones=[self.max_epoch // 2], gamma=0.1)
        return [optimizer], [scheduler]

