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
from dataset import SceneTextDataset

from detect import get_bboxes

from model_lightning import EASTLightningModel
import pytorch_lightning as pl

from argparse import ArgumentParser

from PIL import Image, ImageDraw

import random
import wandb

import shutil

def parse_args():
    parser = ArgumentParser()
    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    
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

def check_dataloader(dataloader):

    save_dir = 'visualize/'

    if os.path.exists(save_dir):  
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)    

    for file in os.listdir():
        if file.endswith('th_aug_image.png'):
            os.remove(file)
            print(f"Deleted: {file}")

    # DataLoader를 통해 증강된 이미지 확인하기
    for batch_idx, (images, score_maps, geo_maps, _) in enumerate(dataloader):

        for image_idx, (image, score_map, geo_map) in enumerate(zip(images, score_maps, geo_maps)):
            bboxes=get_bboxes(score_map.numpy(), geo_map.numpy())
            if bboxes is None:
                bboxes = np.zeros((0, 4, 2), dtype=np.float32)
            else:
                bboxes = bboxes[:, :8].reshape(-1, 4, 2)


            image_sample = image.permute(1, 2, 0).numpy().astype(np.uint8)
            image_sample = Image.fromarray(image_sample)  # [C, H, W] -> [H, W, C]로 변환

            draw = ImageDraw.Draw(image_sample)
            for bbox in bboxes:
                # bbox points
                pts = [(int(p[0]), int(p[1])) for p in bbox]
                draw.polygon(pts, outline=(255, 0, 0))                

            image_name = f'{batch_idx*8+image_idx}th_aug_image.png'
            image_sample.save(os.path.join(save_dir, image_name))
            print(f"Created: {image_name}")

import json

def load_data(data_dir, test_size = 0.2):
    def load_annotations(data_dir):
        lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']

        total_anno = dict()
        for nation in lang_list:
            with open(os.path.join(data_dir, '{}_receipt/ufo/train.json'.format(nation)), 'r', encoding='utf-8') as f:
                anno = json.load(f)
            for im in anno['images']:
                total_anno[im] = anno['images'][im]

        return total_anno
    
    def infer_dir(fname):
        lang_indicator = fname.split('.')[1]
        if lang_indicator == 'zh':
            lang = 'chinese'
        elif lang_indicator == 'ja':
            lang = 'japanese'
        elif lang_indicator == 'th':
            lang = 'thai'
        elif lang_indicator == 'vi':
            lang = 'vietnamese'

        return os.path.join(data_dir, f'{lang}_receipt', 'img', 'train')

    total_anno = load_annotations(data_dir)
    for image_fname in total_anno.keys():
        image_fpath = os.path.join(infer_dir(image_fname), image_fname)
        total_anno[image_fname]['image_path']=image_fpath

    data_list = list(total_anno.values())
    
    random.seed(42)
    random.shuffle(data_list)

    # 전체 데이터 개수
    total_size = len(data_list)

    # 테스트 데이터 개수 계산
    test_size = int(total_size * test_size)

    # 학습 데이터와 테스트 데이터 분리
    train_list = data_list[:-test_size]
    test_list = data_list[-test_size:]

    return train_list, test_list

def main():
    args = parse_args()

    train_list, valid_list = load_data(args.data_dir)

    train_dataset = SceneTextDataset(
            train_list,
            data_type='train',
            image_size=args.image_size,
            crop_size=args.input_size,
        )
    train_dataset = EASTDataset(train_dataset)

    valid_dataset = SceneTextDataset(
            valid_list,
            data_type='valid',
            image_size=args.input_size,
            crop_size=args.input_size,
        )
    valid_dataset = EASTDataset(valid_dataset)
    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)
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