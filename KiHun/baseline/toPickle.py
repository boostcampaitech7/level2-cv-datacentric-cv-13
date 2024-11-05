import pickle
from tqdm import tqdm
import os
import os.path as osp

from east_dataset import EASTDataset
from dataset_new import SceneTextDataset

import albumentations as A

def main():
    data_dir = './data'
    image_size = [2048]
    crop_size = [1024]
    train_dir = f'pickle/{image_size}_cs{crop_size}/train/'
    val_dir = f'pickle/{crop_size}/valid/'
    # 경로 폴더 생성
    os.makedirs(osp.join(data_dir, train_dir), exist_ok=True)
    os.makedirs(osp.join(data_dir, val_dir), exist_ok=True)
    
    for i, i_size in enumerate(image_size):
        for j, c_size in enumerate(crop_size):
            if c_size > i_size:
                continue
            # Create dataset for validation without augmentations
            val_dataset = SceneTextDataset(
                root_dir=data_dir,
                data_type='valid',  # Same dataset but will not apply augmentations
                image_size=c_size,
                crop_size=c_size,
                # You may want to add parameters to disable augmentations if needed
            )
            val_dataset = EASTDataset(val_dataset)

            # Save non-augmented validation dataset
            ds_val = len(val_dataset)
            for k in tqdm(range(ds_val)):
                data = val_dataset.__getitem__(k)  # Get non-augmented data
                with open(file=osp.join(data_dir, val_dir, f"{ds_val*i+ds_val*j+k}.pkl"), mode="wb") as f:
                    pickle.dump(data, f)

            train_dataset = SceneTextDataset(
                    root_dir=data_dir,
                    data_type='train',
                    image_size=i_size,
                    crop_size=c_size,
                )
            train_dataset = EASTDataset(train_dataset)

            ds = len(train_dataset)
            for k in tqdm(range(ds)):
                data = train_dataset.__getitem__(k)
                with open(file=osp.join(data_dir, train_dir, f"{ds*i+ds*j+k}.pkl"), mode="wb") as f:
                    pickle.dump(data, f)

            
if __name__ == '__main__':
    main()