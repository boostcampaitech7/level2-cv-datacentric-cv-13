import os
import numpy as np

from torch.utils.data import DataLoader

from east_dataset import EASTDataset
from dataset_pickle import PickleDataset, load_pickle_files

from model_lightning import EASTLightningModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from argparse import ArgumentParser

from detect import get_bboxes

from utils import visualize_bbox

import wandb

import shutil

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train_dataset_dir', type=str,default="./data/pickle/2048/train")  
    parser.add_argument('--valid_dataset_dir', type=str,default="./data/pickle/1024/valid")  
    # Conventional args
    '''parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))'''

    
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--split_seed', type=int, default=42)

    parser.add_argument('--checkpoint_dir', type=str,default="./trained_models")  

    parser.add_argument('--train_num_workers', type=int, default=8)
    parser.add_argument('--valid_num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--valid_batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    #parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument("--skip_valid", action="store_true", help="skip validation")
    parser.add_argument("--checkaug", action="store_true", help="check aug")
    parser.add_argument("--nowandb", action="store_true", help="disable wandb")

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


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

            visualized_result = visualize_bbox(image, bboxes)
 
            image_name = f'{batch_idx*dataloader.batch_size+image_idx}th_aug_image.png'
            visualized_result.save(os.path.join(save_dir, image_name))
            print(f"Created: {image_name}")

def main():
    args = parse_args()

    #train과 valid 경로의 파일을 각각 나눔
    train_files, valid_files = load_pickle_files(args.train_dataset_dir, args.valid_dataset_dir, args.split_seed, train_ratio = 1.0 if args.skip_valid else 0.8)

    train_pickle = PickleDataset(file_list=train_files, data_type='train', input_image=args.input_size, normalize=not args.checkaug)
    valid_pickle = PickleDataset(file_list=valid_files, data_type='valid', input_image=args.image_size, normalize=not args.checkaug)

    #Dataset 생성
    train_dataset, valid_dataset = EASTDataset(train_pickle), EASTDataset(valid_pickle)

    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers, shuffle=True)
    val_loader = None if args.skip_valid else DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=args.valid_num_workers, shuffle=False)

    if args.checkaug:
        check_dataloader(train_loader)
        return

    model = EASTLightningModel(
        args.image_size, 
        args.input_size, 
        args.learning_rate, 
        args.max_epoch)
    
    wandb_logger = None
    if not args.nowandb:
        config = args.__dict__
        run_name = config.pop('run_name', None)  # 'run_name'이 있으면 가져오고 없으면 None

        wandb_logger = WandbLogger(project='project3_test_run', name=run_name, config=config)

    trainer = pl.Trainer(
        logger=wandb_logger,
        default_root_dir=args.checkpoint_dir,
        max_epochs=args.max_epoch,
        devices=1,
        accelerator="gpu",
        check_val_every_n_epoch=25,
        num_sanity_val_steps=1,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if not args.nowandb:
            wandb.finish()

if __name__ == '__main__':
    main()