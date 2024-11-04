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
from utils import *

def crop_img(img, vertices, labels, length):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices

def resize_img(img, vertices, size):
    h, w = img.height, img.width
    ratio = size / max(h, w)
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    new_vertices = vertices * ratio
    return img, new_vertices

def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, new_vertices

def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices

def pad_if_needed(image, vertices, min_height, min_width, position='top_left', pad_value=(0, 0, 0)):
    # Get the original dimensions
    original_height, original_width = image.size

    # Calculate the padding amounts
    pad_height = max(0, min_height - original_height)
    pad_width = max(0, min_width - original_width)

    # Determine the padding based on the position
    if position == 'top_left':
        top = 0
        left = 0
    elif position == 'top_right':
        top = 0
        left = pad_width
    elif position == 'bottom_left':
        top = pad_height
        left = 0
    elif position == 'bottom_right':
        top = pad_height
        left = pad_width
    elif position == 'center':
        top = pad_height // 2
        left = pad_width // 2
    else:
        raise ValueError("Invalid position. Choose from 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'center'.")

    # Create a new image with padding
    new_width = original_width + pad_width
    new_height = original_height + pad_height
    new_image = Image.new("RGB", (new_width, new_height), pad_value)

    # Paste the original image into the padded image
    new_image.paste(image, (left, top))

    # Update the vertices based on the padding
    updated_vertices = vertices.copy()
    updated_vertices[0::2] += left  # Update x-coordinates
    updated_vertices[1::2] += top   # Update y-coordinates

    return new_image, updated_vertices

def longest_max_size_transform(img, vertices, size):
    h, w = img.height, img.width
    ratio = size / max(h, w)
    
    # Resize the image based on the longest dimension
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    
    # Scale vertices according to the ratio
    new_vertices = vertices * ratio
    return img, new_vertices

def random_scale(image, vertices, scale_range=(0.8, 1.2)):
    # Get the original dimensions
    original_width, original_height = image.size

    # Randomly select a scale factor within the provided range
    scale_factor = random.uniform(scale_range[0], scale_range[1])

    # Calculate the new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image using the calculated dimensions
    scaled_image = image.resize((new_width, new_height), Image.BILINEAR)

    # Adjust the vertices by the scale factor
    updated_vertices = vertices * scale_factor

    return scaled_image, updated_vertices