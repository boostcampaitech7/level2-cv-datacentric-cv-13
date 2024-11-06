import math
import numpy as np
from numba import njit
from PIL import Image, ImageDraw

@njit
def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@njit
def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices

@njit
def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
    r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
    r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
    r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
        offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v

@njit
def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

@njit
def rotate_vertices(vertices, angle, center):
    cos_theta, sin_theta = math.cos(angle), math.sin(angle)
    rotated_vertices = np.zeros(vertices.shape)
    for i in range(0, len(vertices), 2):
        x, y = vertices[i] - center[0], vertices[i + 1] - center[1]
        rotated_vertices[i] = x * cos_theta - y * sin_theta + center[0]
        rotated_vertices[i + 1] = x * sin_theta + y * cos_theta + center[1]
    return rotated_vertices

@njit
def rotate_vertices_numba(vertices, angle, center):
    """Rotates vertices around a given center point.

    Args:
        vertices: A numpy array of shape (N, 8) representing N vertices.
        angle: Rotation angle in radians.
        center: A numpy array of shape (2,) representing the center point.

    Returns:
        A numpy array of shape (N, 8) representing the rotated vertices.
    """

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cx, cy = center

    rotated_vertices = np.zeros_like(vertices)
    for i in range(vertices.shape[0]):
        x, y = vertices[i][::2], vertices[i][1::2]
        dx, dy = x - cx, y - cy
        rotated_x = dx * cos_angle - dy * sin_angle + cx
        rotated_y = dx * sin_angle + dy * cos_angle + cy
        rotated_vertices[i][::2] = rotated_x
        rotated_vertices[i][1::2] = rotated_y

    return rotated_vertices

@njit
def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

@njit
def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err

@njit
def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi

@njit
def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y
    
# def generate_roi_mask(image, vertices, labels):
#     mask = np.ones(image.shape[:2], dtype=np.float32)
#     ignored_polys = []
#     for vertice, label in zip(vertices, labels):
#         if label == 0:
#             ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
#     cv2.fillPoly(mask, ignored_polys, 0)
#     return mask

@njit
def generate_roi_mask(image: np.ndarray, vertices: np.ndarray, labels: np.ndarray):
    mask = np.ones(image.shape[:2], dtype=np.float32)

    for vertice, label in zip(vertices, labels):
        if label == 0:
            fill_polygon(mask, vertice, 0)

    return mask

@njit
def fill_polygon(mask, polygon, value):
    # 다각형의 bounding box를 계산합니다.
    min_x, max_x = np.min(polygon[::2]), np.max(polygon[::2])
    min_y, max_y = np.min(polygon[1::2]), np.max(polygon[1::2])

    # bounding box 내의 각 점이 다각형 내부에 있는지 확인
    for y in range(max(0, min_y), min(mask.shape[0], max_y + 1)):
        for x in range(max(0, min_x), min(mask.shape[1], max_x + 1)):
            if point_in_polygon(x, y, polygon):
                mask[y, x] = value

@njit
def point_in_polygon(x, y, polygon):
    # point-in-polygon (PIP) 알고리즘: Ray-casting 기법 사용
    num_points = polygon.shape[0] // 2  # 꼭짓점 개수를 2로 나누어 계산
    j = num_points - 1
    odd_nodes = False
    for i in range(num_points):
        xi, yi = polygon[2*i], polygon[2*i+1]
        xj, yj = polygon[2*j], polygon[2*j+1]
        if ((yi < y <= yj) or (yj < y <= yi)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            odd_nodes = not odd_nodes
        j = i
    return odd_nodes

@njit
def is_cross_text_bounding_box(start_loc, length, vertices):
    """
    Bounding Box를 이용하여 자유사변형과 사각형이 교차하는지 확인

    Args:
        start_loc (tuple): 사각형의 시작 좌표
        length (int): 사각형의 한 변의 길이
        vertices (np.ndarray): 자유사변형의 꼭짓점 좌표 (Nx2 배열)

    Returns:
        bool: 교차하면 True, 아니면 False
    """

    # 사각형의 Bounding Box 계산
    rect_min_x, rect_min_y = start_loc
    rect_max_x = rect_min_x + length
    rect_max_y = rect_min_y + length

    # 자유사변형의 Bounding Box 계산
    poly_min_x = np.min(vertices[:, 0])
    poly_max_x = np.max(vertices[:, 0])
    poly_min_y = np.min(vertices[:, 1])
    poly_max_y = np.max(vertices[:, 1])

    # 두 Bounding Box가 겹치는지 확인
    return (rect_min_x <= poly_max_x and poly_min_x <= rect_max_x) and \
           (rect_min_y <= poly_max_y and poly_min_y <= rect_max_y)

@njit
def polygon_area(vertices):
    """
    2D 자유사변형의 넓이를 삼각형 분할 및 벡터 외적을 이용하여 계산

    Args:
        vertices (np.ndarray): 자유사변형의 꼭짓점 좌표 (Nx8 배열)

    Returns:
        np.ndarray: 각 자유사변형의 넓이 (Nx1 배열)
    """

    n = vertices.shape[0]
    areas = np.zeros(n)

    for i in range(n):
        # 꼭짓점 좌표를 reshape
        v = vertices[i].reshape(4, 2)

        # 삼각형 분할 (첫 번째 점을 기준으로)
        for j in range(1, 3):
            # 두 변의 벡터 계산
            vec1 = v[j] - v[0]
            vec2 = v[j+1] - v[0]

            # 벡터 외적 (2D에서 z 성분만 계산)
            cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]

            # 삼각형 넓이 계산 (절댓값)
            area = 0.5 * np.abs(cross_product)
            areas[i] += area

    return areas

@njit
def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()
    areas = polygon_area(vertices)

    labels[areas < ignore_under] = 0
    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels

def visualize_bbox(img, bboxes):
    img = img.permute(1, 2, 0).numpy().astype(np.uint8)
    img = Image.fromarray(img)  # [C, H, W] -> [H, W, C]로 변환

    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        # bbox points
        pts = [(int(p[0]), int(p[1])) for p in bbox]
        draw.polygon(pts, outline=(255, 0, 0))

    return img