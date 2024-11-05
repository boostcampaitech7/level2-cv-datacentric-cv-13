import os
import torch
import numpy as np
import cv2

import math
import matplotlib.pyplot as plt

from torch import cuda
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Precision, Recall, F1Score  # 추가된 torchmetrics

from east_dataset import EASTDataset
from dataset import SceneTextDataset, PickleDataset

from model_lightning import EASTLightningModel
import pytorch_lightning as pl

from argparse import ArgumentParser

import random
import wandb

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train_dataset_dir', type=str,default="./data/pickle/2048/")  
    parser.add_argument('--valid_dataset_dir', type=str,default="./data/pickle/1024/")  
    # Conventional args
    '''parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))'''

    
    parser.add_argument('--checkpoint_dir', type=str,default="./trained_models")  

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    #parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument("--amp", action="store_true", help="Enable AMP")
    parser.add_argument("--checkaug", action="store_true", help="check aug")
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

    train_pickle = PickleDataset(file_list=train_files, data_type='train')
    valid_pickle = PickleDataset(file_list=valid_files, data_type='valid')

    return train_pickle, valid_pickle

def check_dataloader(dataloader):

    for file in os.listdir():
        if file.endswith('th_aug_image.png'):
            os.remove(file)
            print(f"Deleted: {file}")

    # DataLoader를 통해 증강된 이미지 확인하기
    for batch_idx, (images, score_maps, geo_maps, roi_masks) in enumerate(dataloader):

        for image_idx, (image, score_map, geo_map, roi_mask) in enumerate(zip(images, score_maps, geo_maps, roi_masks)):
            image_sample = image.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]로 변환
            image_sample = (image_sample * 0.5 + 0.5).clip(0, 1)

            # 마스크 윤곽선 강조
            contours, hierarchy = cv2.findContours(roi_mask.permute(1,2,0).numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_sample, contours, -1, (0, 255, 0), 2)  # 초록색으로 윤곽선 표시

            image_name = f'{batch_idx*8+image_idx}th_aug_image.png'

            plt.imsave(image_name, image_sample)
            print(f"Created: {image_name}")

def main():
    args = parse_args()

    torch.cuda.empty_cache()

    # train과 valid 경로의 파일을 각각 나눔
    train_pickle, valid_pickle = load_pickle_files(args.train_dataset_dir, args.valid_dataset_dir)

    # Dataset 생성
    train_dataset = EASTDataset(train_pickle)
    valid_dataset = EASTDataset(valid_pickle)

    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    if args.checkaug:
        check_dataloader(train_loader)
        return

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
        check_val_every_n_epoch=5,
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