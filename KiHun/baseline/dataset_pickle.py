import pickle
from tqdm import tqdm
import os

from torch.utils.data import Dataset
from dataset import SceneTextDataset

from aug import *
from utils import *

import albumentations as A

class PickleDataset(Dataset):
    def __init__(self, file_list, data_type, input_image, normalize=True,
                 ignore_under_threshold=10,
                 drop_under_threshold=1,):
        self.file_list = file_list
        self.data_type = data_type
        self.input_image = input_image
        self.normalize = normalize

        self.config_filter_vertices = [ignore_under_threshold, drop_under_threshold]

    def _convert_to_rgb_numpy(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        return image
 
    def _train_augmentation(self, image, vertices, labels):
        image = self._convert_to_rgb_numpy(image)
        
        image, vertices = random_scale(image, vertices, scale_range=(0.5, 1.0))
        image, vertices = rotate_img(image, vertices)
        image, vertices, labels = crop_img2(image, vertices, labels, self.input_image)

        vertices, labels = filter_vertices(
            vertices,labels,
            ignore_under=self.config_filter_vertices[0],
            drop_under=self.config_filter_vertices[1]
        )

        image, vertices, labels = generate_lines(image, vertices, labels)

        transform = [A.ColorJitter(hue=(-0.05,0.05), brightness=(0.75, 1.25), contrast=(0.75, 1.25), saturation=(0.6, 0.75))]
        # transform 적용
        if self.normalize is True:
            transform.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        image = A.Compose(transform)(image=image)['image']

        return image, vertices, labels

    def _valid_augmentation(self, image, vertices, labels):

        image = self._convert_to_rgb_numpy(image)

        vertices = vertices.astype(np.float64)

         # transform 적용
        if self.normalize is True:
            transform = A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
            image = transform(image=image)['image']
       

        return image, vertices, labels

    def __getitem__(self, idx):

        # 각 파일을 불러오고 필요한 전처리 및 증강 적용
        file_path = self.file_list[idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        image, vertices, labels = data

        if self.data_type == 'train':
            image, vertices, labels = self._train_augmentation(image, vertices, labels)
        else:
            image, vertices, labels = self._valid_augmentation(image, vertices, labels)

        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask

    def __len__(self):
        return len(self.file_list)

def load_pickle_files(train_dir, valid_dir, split_seed, total_files=400, train_ratio=0.8):
    # 0부터 399까지의 파일 인덱스를 생성하고 무작위로 섞기
    indices = list(range(0, total_files))

    random.seed(split_seed)
    random.shuffle(indices)

    # 80%는 train, 20%는 valid로 할당
    train_indices = indices[:int(total_files * train_ratio)]
    valid_indices = indices[int(total_files * train_ratio):]

    # 파일 경로 생성
    train_files = [os.path.join(train_dir, f"{idx}.pkl") for idx in train_indices]
    valid_files = [os.path.join(valid_dir, f"{idx}.pkl") for idx in valid_indices]

    return train_files, valid_files

def createPickles(data_dir, data_type, image_size):
    train_dir = f'pickle/{image_size}/{data_type}/'
    # 경로 폴더 생성
    os.makedirs(os.path.join(data_dir, train_dir), exist_ok=True)
    
    dataset = SceneTextDataset(
            root_dir=data_dir,
            data_type=data_type,
            image_size=image_size,
        )

    ds = len(dataset)
    for k in tqdm(range(ds)):
        data = dataset.__getitem__(k)
        with open(file=os.path.join(data_dir, train_dir, f"{k}.pkl"), mode="wb") as f:
            pickle.dump(data, f)

def main():
    data_dir = './data'

    #createPickles(data_dir, 'valid', 1024)
    createPickles(data_dir, 'train', 2048)
    
        
if __name__ == '__main__':
    main()