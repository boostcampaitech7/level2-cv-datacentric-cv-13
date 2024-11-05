import pickle
from tqdm import tqdm
import os
import os.path as osp

from dataset import SceneTextDataset

import albumentations as A

def createPickles(data_dir, image_size):
    train_dir = f'pickle/{image_size}/'
    # 경로 폴더 생성
    os.makedirs(osp.join(data_dir, train_dir), exist_ok=True)
    
    train_dataset = SceneTextDataset(
            root_dir=data_dir,
            data_type='train',
            image_size=image_size,
        )

    ds = len(train_dataset)
    for k in tqdm(range(ds)):
        data = train_dataset.__getitem__(k)
        with open(file=osp.join(data_dir, train_dir, f"{k}.pkl"), mode="wb") as f:
            pickle.dump(data, f)

def main():
    data_dir = './data'

    createPickles(data_dir, 1024)
    createPickles(data_dir, 2048)
    
        
if __name__ == '__main__':
    main()