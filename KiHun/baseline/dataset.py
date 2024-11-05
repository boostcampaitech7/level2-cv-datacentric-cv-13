import os.path as osp
import json
from PIL import Image

import numpy as np
from torch.utils.data import Dataset

from aug import *
from utils import *

class SceneTextDataset(Dataset):
    def __init__(self, root_dir,
                 data_type='train',
                 image_size=2048,
                 crop_size=1024,
                 ignore_under_threshold=10,
                 drop_under_threshold=1,
                 color_jitter=False,
                 normalize=True):
        self._lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
        self.root_dir = root_dir
        self.data_type = data_type

        if data_type == 'test':
            split = 'test'
        else:
            split = 'train'

        total_anno = dict(images=dict())
        for nation in self._lang_list:
            with open(osp.join(root_dir, '{}_receipt/ufo/{}.json'.format(nation, split)), 'r', encoding='utf-8') as f:
                anno = json.load(f)
            for im in anno['images']:
                total_anno['images'][im] = anno['images'][im]

        self.anno = total_anno
        self.image_fnames = sorted(self.anno['images'].keys())

        self.image_size, self.crop_size = image_size, crop_size
        self.color_jitter, self.normalize = color_jitter, normalize

        self.drop_under_threshold = drop_under_threshold
        self.ignore_under_threshold = ignore_under_threshold

    def _infer_dir(self, fname):
        lang_indicator = fname.split('.')[1]
        if lang_indicator == 'zh':
            lang = 'chinese'
        elif lang_indicator == 'ja':
            lang = 'japanese'
        elif lang_indicator == 'th':
            lang = 'thai'
        elif lang_indicator == 'vi':
            lang = 'vietnamese'
        else:
            raise ValueError

        if self.data_type == 'test':
            split = 'test'
        else:
            split = 'train'

        return osp.join(self.root_dir, f'{lang}_receipt', 'img', split)

    def __len__(self):
        return len(self.image_fnames)

    def _load_annotations(self, image_fname):    
        vertices, labels = [], []
        for word_info in self.anno['images'][image_fname]['words'].values():
            num_pts = np.array(word_info['points']).shape[0]
            if num_pts > 4:
                continue
            vertices.append(np.array(word_info['points']).flatten())
            labels.append(1)
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)
        return vertices, labels

    def _preprocess(self, image, vertices, labels):
        image, vertices = crop_img_custom(image, vertices)
        vertices, labels = filter_vertices(
            vertices,labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )
        image, vertices = longest_max_size_transform(image, vertices, self.image_size)
        image, vertices = pad_if_needed(image, vertices, min_height = self.image_size, min_width = self.image_size, pad_value=(0,0,0))

        image, vertices = crop_img(image, vertices, labels, self.image_size)

        return image, vertices, labels

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self._infer_dir(image_fname), image_fname)

        vertices, labels = self._load_annotations(image_fname)
        image = Image.open(image_fpath)

        image, vertices, labels = self._preprocess(image, vertices, labels)

        return image, vertices, labels