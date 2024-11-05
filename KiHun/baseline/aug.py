import math
from PIL import Image

import random

import cv2
import numpy as np
from numba import njit

from utils import rotate_vertices, is_cross_text_bounding_box

def crop_img_custom(image, vertices):

    x_coords = vertices[:,[0,2,4,6]].flatten()
    y_coords = vertices[:,[1,3,5,7]].flatten()

    # 최소 및 최대 x, y 좌표 계산
    min_x, min_y = np.min(x_coords), np.min(y_coords)
    max_x, max_y = np.max(x_coords), np.max(y_coords)
    
    h, w = image.size[1], image.size[0]  
    crop_w, crop_h = max_x - min_x, max_y - min_y
    
    # 이미지 크기에 따라 여백 비율 조정
    width_margin = int((w - crop_w) * 0.05)  # 너비 여백을 이미지 너비의 5%로 설정
    height_margin = int((h - crop_h) * 0.1)  # 높이 여백을 이미지 높이의 10%로 설정

    # 여백을 포함하여 새로운 크롭 좌표 설정
    new_min_x = max(0, min_x - width_margin)
    new_max_x = min(w, max_x + width_margin)
    new_min_y = max(0, min_y - height_margin)
    new_max_y = min(h, max_y + height_margin)

    image = image.crop([new_min_x, new_min_y, new_max_x, new_max_y])

    new_vertices = vertices.copy()  # vertices 배열을 복사하여 새로운 배열에 저장
    for idx, vertice in enumerate(vertices):
        for i in range(0, len(vertice), 2):
            new_vertices[idx][i] -= new_min_x  # x 좌표에서 new_min_x만큼 뺌
            new_vertices[idx][i+1] -= new_min_y  # y 좌표에서 new_min_y만큼 뺌

    return image, new_vertices

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

def pad_if_needed(image, vertices, min_height, min_width, position='center', pad_value=(0, 0, 0)):
    # Get the original dimensions
    original_height = image.height
    original_width = image.width

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
    for idx, vertice in enumerate(vertices):
        for i in range(0, len(vertice), 2):
            updated_vertices[idx][i] += left  # Update x-coordinates
            updated_vertices[idx][i+1] += top   # Update y-coordinates

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

@njit
def crop_img(image: np.ndarray, vertices: np.ndarray, labels: np.ndarray, length: int):
    h, w = image.shape[:2]

    # 이미지 크기가 지정한 길이 이상이 되도록 조정
    if h >= w and w < length:
        new_w, new_h = length, int(h * length / w)
    elif h < w and h < length:
        new_w, new_h = int(w * length / h), length
    else:
        new_w, new_h = w, h

    # NumPy로 크기 조정: 선형 보간을 직접 계산
    resized_image = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    scale_x, scale_y = w / new_w, h / new_h
    for i in range(new_h):
        for j in range(new_w):
            resized_image[i, j] = image[int(i * scale_y), int(j * scale_x)]
    
    ratio_w, ratio_h = new_w / w, new_h / h

    # vertices의 x, y 좌표에 각각 ratio_w, ratio_h 적용
    new_vertices = vertices.copy()
    for i in range(vertices.shape[0]):
        # x 좌표들에 대해 ratio_w 적용
        new_vertices[i, 0] *= ratio_w  # x1
        new_vertices[i, 2] *= ratio_w  # x2
        new_vertices[i, 4] *= ratio_w  # x3
        new_vertices[i, 6] *= ratio_w  # x4
        # y 좌표들에 대해 ratio_h 적용
        new_vertices[i, 1] *= ratio_h  # y1
        new_vertices[i, 3] *= ratio_h  # y2
        new_vertices[i, 5] *= ratio_h  # y3
        new_vertices[i, 7] *= ratio_h  # y4

    # 크롭할 영역 위치 선택
    remain_w, remain_h = new_w - length, new_h - length
    for _ in range(1000):
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        if not is_cross_text_bounding_box([start_w, start_h], length, new_vertices[labels == 1]):
            break
    cropped_image = resized_image[start_h:start_h + length, start_w:start_w + length]

    # vertices를 크롭된 이미지 기준으로 변환
    for i in range(new_vertices.shape[0]):
        new_vertices[i, 0] -= start_w  # x1
        new_vertices[i, 2] -= start_w  # x2
        new_vertices[i, 4] -= start_w  # x3
        new_vertices[i, 6] -= start_w  # x4
        new_vertices[i, 1] -= start_h  # y1
        new_vertices[i, 3] -= start_h  # y2
        new_vertices[i, 5] -= start_h  # y3
        new_vertices[i, 7] -= start_h  # y4

    return cropped_image, new_vertices

@njit
def rotate_img(image: np.ndarray, vertices: np.ndarray, angle_range=10):
    h, w = image.shape[:2]
    angle = angle_range * (np.random.rand() * 2 - 1)
    radian = angle * np.pi / 180
    cos_theta, sin_theta = math.cos(radian), math.sin(radian)
    center_x, center_y = (w - 1) / 2, (h - 1) / 2

    # 회전 변환을 적용한 좌표 생성
    rotated_image = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            x = cos_theta * (j - center_x) - sin_theta * (i - center_y) + center_x
            y = sin_theta * (j - center_x) + cos_theta * (i - center_y) + center_y
            if 0 <= x < w and 0 <= y < h:
                rotated_image[i, j] = image[int(y), int(x)]

    # vertices 회전 적용
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i, :] = rotate_vertices(vertice, -radian, np.array([center_x, center_y]))
    return rotated_image, new_vertices

@njit
def random_scale(image: np.ndarray, vertices: np.ndarray, scale_range=(0.8, 1.2)):
    h, w = image.shape[:2]
    scale_factor = random.uniform(*scale_range)
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    # NumPy로 크기 조정: 선형 보간을 직접 계산
    scaled_image = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    for i in range(new_h):
        for j in range(new_w):
            x = min(int(i / scale_factor), h - 1)
            y = min(int(j / scale_factor), w - 1)
            scaled_image[i, j] = image[x, y]
            
    updated_vertices = vertices * scale_factor

    return scaled_image, updated_vertices