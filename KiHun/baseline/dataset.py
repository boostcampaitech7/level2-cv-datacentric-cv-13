import os.path as osp
import math
import json
from PIL import Image

import pickle
import random
import os

import torch
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from numba import njit
from aug import *
from utils import *

class SceneTextDataset(Dataset):
    def __init__(self,
                 data_list,
                 data_type = 'train',
                 image_size=2048,
                 crop_size=1024,
                 ignore_under_threshold=10,
                 drop_under_threshold=1,
                 color_jitter=True,
                 normalize=True):
        

        self.data_list = data_list
        self.data_type = data_type

        self.image_size, self.crop_size = image_size, crop_size
        self.color_jitter, self.normalize = color_jitter, normalize

        self.drop_under_threshold = drop_under_threshold
        self.ignore_under_threshold = ignore_under_threshold

    def __len__(self):
        return len(self.data_list)

    def _load_annotations(self, item):    
        vertices, labels = [], []
        for word_info in item['words'].values():
            num_pts = np.array(word_info['points']).shape[0]
            if num_pts > 4:
                continue
            vertices.append(np.array(word_info['points']).flatten())
            labels.append(1)
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)
        return vertices, labels

    def _train_preprocess(self, image, vertices, labels):

        #image, vertices = resize_img(image, vertices, self.image_size)
        #image, vertices = adjust_height(image, vertices)
        
        image, vertices = crop_img_custom(image, vertices)
        vertices, labels = filter_vertices(
            vertices,labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )
        image, vertices = longest_max_size_transform(image, vertices, self.image_size)
        image, vertices = pad_if_needed(image, vertices, min_height = self.image_size, min_width = self.image_size, pad_value=(0,0,0))
        image, vertices = random_scale(image, vertices, scale_range=(0.6, 0.75))
        vertices, labels = filter_vertices(
            vertices,labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )
        image, vertices = rotate_img(image, vertices)
        image, vertices = crop_img(image, vertices, labels, self.crop_size)



        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        funcs = []
        if self.color_jitter:
            funcs.append(A.ColorJitter(hue=(0, 0)))
        if self.normalize:
            funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = A.Compose(funcs)

        image = transform(image=image)['image']

        return image, vertices, labels

    def _valid_preprocess(self, image, vertices, labels):
        image, vertices = longest_max_size_transform(image, vertices, self.image_size)
        image, vertices = pad_if_needed(image, vertices, min_height = self.image_size, min_width = self.image_size, pad_value=(0,0,0))

        image, vertices = crop_img(image, vertices, labels, self.image_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        transform = A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        image = transform(image=image)['image']

        return image, vertices, labels

    def __getitem__(self, idx):
        item = self.data_list[idx]

        vertices, labels = self._load_annotations(item)
        image = Image.open(item['image_path'])

        if self.data_type == 'train':
            image, vertices, labels = self._train_preprocess(image, vertices, labels)
        else:
            image, vertices, labels = self._valid_preprocess(image, vertices, labels)

        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask
