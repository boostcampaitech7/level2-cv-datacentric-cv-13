import os
import torch
import numpy as np

import math
import pytorch_lightning as pl

from torch import cuda
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Precision, Recall, F1Score  # 추가된 torchmetrics

from east_dataset import EASTDataset
from dataset_new import SceneTextDataset, PickleDataset

from model import EAST

from datetime import timedelta
from argparse import ArgumentParser

import wandb

from deteval import calc_deteval_metrics
from detect import get_bboxes

import random

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train_dataset_dir', type=str,default="/data/ephemeral/home/code/data/pickle/[2048]_cs[1024]/train/")  
    parser.add_argument('--valid_dataset_dir', type=str,default="/data/ephemeral/home/code/data/pickle/[1024]/valid/")  
    # Conventional args
    '''parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))'''

    
    parser.add_argument('--checkpoint_dir', type=str,default="./code/trained_models")  

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    #parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument("--amp", action="store_true", help="Enable AMP")
    parser.add_argument("--nowandb", action="store_true", help="disable wandb")

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def wandb_setup(config):

    wandb.init(
        project="project3_test_run",
        config=config
    )

def load_pickle_files(train_dir, valid_dir, total_files=400, train_ratio=0.8):
    # 0부터 399까지의 파일 인덱스를 생성하고 무작위로 섞기
    indices = list(range(0, total_files))

    random.seed(42)
    random.shuffle(indices)

    # 80%는 train, 20%는 valid로 할당
    train_indices = indices[:int(total_files * train_ratio)]
    valid_indices = indices[int(total_files * train_ratio):]

    # 파일 경로 생성
    train_files = [os.path.join(train_dir, f"{idx}.pkl") for idx in train_indices]
    valid_files = [os.path.join(valid_dir, f"{idx}.pkl") for idx in valid_indices]

    return train_files, valid_files

class EASTLightningModel(pl.LightningModule):
    def __init__(self, image_size, input_size, learning_rate, max_epoch, nowandb):
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


def main():
    args = parse_args()

    torch.cuda.empty_cache()

    # train과 valid 경로의 파일을 각각 나눔
    train_files, valid_files = load_pickle_files(args.train_dataset_dir, args.valid_dataset_dir)

    # Dataset 생성
    train_dataset = PickleDataset(train_files)
    valid_dataset = PickleDataset(valid_files)
    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4, shuffle=False)

    model = EASTLightningModel(
        args.image_size, 
        args.input_size, 
        args.learning_rate, 
        args.max_epoch,
        args.nowandb)
    
    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        max_epochs=args.max_epoch,
        devices=1,
        accelerator="gpu",
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        precision=16 if args.amp else 32  # AMP를 위한 precision 설정
    )

    if not args.nowandb:
        wandb_setup(args.__dict__)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if not args.nowandb:
            wandb.finish()

if __name__ == '__main__':
    main()