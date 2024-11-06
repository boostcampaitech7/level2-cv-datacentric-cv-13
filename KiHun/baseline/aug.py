import math
from PIL import Image

import random

import cv2
import numpy as np
from numba import njit

from utils import rotate_vertices, is_cross_text_bounding_box, fill_quadrilateral_with_radial_gradient, sample_outer_colors

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
def crop_img2(image: np.ndarray, vertices: np.ndarray, labels: np.ndarray, length: int):
    h, w = image.shape[:2]

    # 이미지 크기가 지정한 길이 이상이 되도록 조정
    if h >= w and w < length:
        new_w, new_h = length, int(h * length / w)
    elif h < w and h < length:
        new_w, new_h = int(w * length / h), length
    else:
        new_w, new_h = w, h

    # 이미지 크기 조정
    resized_image = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    scale_x, scale_y = w / new_w, h / new_h
    for i in range(new_h):
        for j in range(new_w):
            resized_image[i, j] = image[int(i * scale_y), int(j * scale_x)]
    
    ratio_w, ratio_h = new_w / w, new_h / h

    # vertices 크기 조정
    new_vertices = vertices.copy()
    for i in range(vertices.shape[0]):
        new_vertices[i, 0::2] *= ratio_w  # x 좌표들
        new_vertices[i, 1::2] *= ratio_h  # y 좌표들

    # 크롭할 영역 위치 선택
    remain_w, remain_h = new_w - length, new_h - length

    start_w = int(np.random.rand() * remain_w)
    start_h = int(np.random.rand() * remain_h)

    # 이미지 크롭
    cropped_image = resized_image[start_h:start_h + length, start_w:start_w + length]

    # vertices를 크롭된 이미지 기준으로 변환
    for i in range(new_vertices.shape[0]):
        new_vertices[i, 0::2] -= start_w  # x 좌표들
        new_vertices[i, 1::2] -= start_h  # y 좌표들

        # vertices가 크롭 영역을 벗어나면 label을 0으로 설정
        if (np.any(new_vertices[i, 0::2] < 0) or np.any(new_vertices[i, 0::2] >= length) or
            np.any(new_vertices[i, 1::2] < 0) or np.any(new_vertices[i, 1::2] >= length)):
            labels[i] = 0

    return cropped_image, new_vertices, labels

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
        flag = is_cross_text_bounding_box([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices

@njit
def rotate_img(image: np.ndarray, vertices: np.ndarray, angle_range=15):
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

    # vertices 회전 적용 및 필터링
    new_vertices = np.zeros(vertices.shape)
    for i in range(vertices.shape[0]):
        new_vertices[i] = rotate_vertices(vertices[i], -radian, np.array([center_x, center_y]))

    # 유효한 vertices만 반환
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

from PIL import ImageDraw

def remove_separator(img, vertices, labels):
    """
    labels가 2인 경우, 해당 사변형 영역을 그라데이션으로 덮어줍니다.
    """
    img_np = np.array(img)
    draw = ImageDraw.Draw(img)

    # 전체 이미지의 밝기 히스토그램을 분석하여 밝기 임계값 설정
    hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    brightness_values = hsv_img[:, :, 2].flatten()  # V 채널 값
    brightness_threshold = np.percentile(brightness_values, 50)  # 상위 50% 밝기 기준

    for idx, label in enumerate(labels):
        if label == 2:  # separator인 경우
            quad_vertices = vertices[idx]
            # 각 꼭지점 주변의 바깥쪽 색상을 샘플링하여 평균을 구합니다.
            sampled_colors = sample_outer_colors(img_np, quad_vertices, brightness_threshold)
            # 그라데이션으로 사변형을 채웁니다.
            fill_quadrilateral_with_radial_gradient(draw, quad_vertices, sampled_colors)

    return img, vertices

import colorsys

def generate_lines(image_np, vertices, labels, line_type='random', thickness=(1,5), gap=(0,4)):

    def draw_line(image_np, line_type, gap, point, color):

        start_point, end_point = point

        if line_type == 'dotted':
            gap += 1
            for i in range(int(np.linalg.norm(np.subtract(end_point, start_point)) / gap)):
                pos = (
                    int(start_point[0] + i * gap * (end_point[0] - start_point[0]) / np.linalg.norm(np.subtract(end_point, start_point))),
                    int(start_point[1] + i * gap * (end_point[1] - start_point[1]) / np.linalg.norm(np.subtract(end_point, start_point)))
                )
                cv2.circle(image_np, pos, thickness // 2, color, -1)
        elif line_type == 'dashed': 
            gap += 8
            for i in range(0, int(np.linalg.norm(np.subtract(end_point, start_point)) / gap), 2):
                pos1 = (
                    int(start_point[0] + i * gap * (end_point[0] - start_point[0]) / np.linalg.norm(np.subtract(end_point, start_point))),
                    int(start_point[1] + i * gap * (end_point[1] - start_point[1]) / np.linalg.norm(np.subtract(end_point, start_point)))
                )
                pos2 = (
                    int(start_point[0] + (i + 1) * gap * (end_point[0] - start_point[0]) / np.linalg.norm(np.subtract(end_point, start_point))),
                    int(start_point[1] + (i + 1) * gap * (end_point[1] - start_point[1]) / np.linalg.norm(np.subtract(end_point, start_point)))
                )
                cv2.line(image_np, pos1, pos2, color, thickness)
        else:
            cv2.line(image_np, start_point, end_point, color, thickness)

        return image_np

    def random_color(h=(0,360), s=(0,100), v=(0,100)):
        """어두운 계통의 랜덤 색상을 생성하여 RGB 값으로 반환하는 함수"""
        
        # HSV 값 설정 (V는 0~20, S는 0~20, H는 0~360 랜덤)
        h = random.uniform(h[0], h[1]) / 360
        s = random.uniform(s[0], s[1]) / 100  # S를 0~1 범위로 조정
        v = random.uniform(v[0], v[1]) / 100  # V를 0~1 범위로 조정
        
        # HSV를 RGB로 변환
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # RGB 값을 0~255 범위로 조정하여 반환
        return int(r*255), int(g*255), int(b*255)
    
    thickness = random.randint(thickness[0], thickness[1])
    line_type_list = ['dotted', 'dashed', 'line']

    for vertice, label in zip(vertices, labels):
        if label == 2:
            points = vertice.reshape(-1,2)

            start_point = int((points[0][0] + points[3][0])//2), int((points[0][1]+points[3][1])//2)
            end_point = int((points[1][0] + points[2][0])//2), int((points[1][1]+points[2][1])//2)

            line_type_per_line = line_type
            if line_type_per_line == 'random':
                line_type_per_line = random.choice(line_type_list)     

            draw_line(image_np = image_np,
                    line_type = line_type_per_line,
                    gap = random.randint(gap[0], gap[1]),
                    point = (start_point, end_point),
                    color = random_color(s=(0,20), v=(0,20)))
        
    return image_np