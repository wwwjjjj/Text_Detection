from PIL import Image,ImageDraw
import numpy as np
import math
import torch
import pyclipper
import cv2
import Polygon as plg
from scipy.ndimage.morphology import distance_transform_edt
def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image

def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))
def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))
def find_bottom(pts):

    if len(pts) > 4:
        e = np.concatenate([pts, pts[:3]])
        candidate = []
        for i in range(1, len(pts) + 1):
            v_prev = e[i] - e[i - 1]
            v_next = e[i + 2] - e[i + 1]
            if cos(v_prev, v_next) < -0.7:
                candidate.append((i % len(pts), (i + 1) % len(pts), norm2(e[i] - e[i + 1])))

        if len(candidate) != 2 or candidate[0][0] == candidate[1][1] or candidate[0][1] == candidate[1][0]:
            # if candidate number < 2, or two bottom are joined, select 2 farthest edge
            mid_list = []
            for i in range(len(pts)):
                mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                mid_list.append((i, (i + 1) % len(pts), mid_point))

            dist_list = []
            for i in range(len(pts)):
                for j in range(len(pts)):
                    s1, e1, mid1 = mid_list[i]
                    s2, e2, mid2 = mid_list[j]
                    dist = norm2(mid1 - mid2)
                    dist_list.append((s1, e1, s2, e2, dist))
            bottom_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-2:]
            bottoms = [dist_list[bottom_idx[0]][:2], dist_list[bottom_idx[1]][:2]]
        else:
            bottoms = [candidate[0][:2], candidate[1][:2]]

    else:
        d1 = norm2(pts[1] - pts[0]) + norm2(pts[2] - pts[3])
        d2 = norm2(pts[2] - pts[1]) + norm2(pts[0] - pts[3])
        bottoms = [(0, 1), (2, 3)] if d1 < d2 else [(1, 2), (3, 0)]
    assert len(bottoms) == 2, 'fewer than 2 bottoms'
    return bottoms
def find_long_edges(points, bottoms):
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)
    i = (b1_end + 1) % n_pts
    long_edge_1 = []

    while (i % n_pts != b2_end):
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_1.append((start, end))
        i = (i + 1) % n_pts

    i = (b2_end + 1) % n_pts
    long_edge_2 = []
    while (i % n_pts != b1_end):
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_2.append((start, end))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2

def judge(x,L):
    if x>=0 and x<L:
        return True
    return False
def split_edge_seqence(points, long_edge, n_parts):

    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while(cur_node<len(point_cumsum) and cur_end > point_cumsum[cur_node + 1]):
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)

def generate_gt(boxes, dim=(512, 512)):
    top_left = []
    top_right = []
    bottom_right = []
    bottom_left = []
    seg = np.zeros((4, dim[0], dim[1]))

    if boxes.size > 0:
        top_left_mask = Image.new('L', (dim[1], dim[0]))
        top_left_draw = ImageDraw.Draw(top_left_mask)
        top_right_mask = Image.new('L', (dim[1], dim[0]))
        top_right_draw = ImageDraw.Draw(top_right_mask)
        bottom_right_mask = Image.new('L', (dim[1], dim[0]))
        bottom_right_draw = ImageDraw.Draw(bottom_right_mask)
        bottom_left_mask = Image.new('L', (dim[1], dim[0]))
        bottom_left_draw = ImageDraw.Draw(bottom_left_mask)
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, x3, y3, x4, y4 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], boxes[i][4], boxes[i][5], boxes[i][6], boxes[i][7]
            ## get box
            side1 = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
            side2 = math.sqrt(math.pow(x3 - x2, 2) + math.pow(y3 - y2, 2))
            side3 = math.sqrt(math.pow(x4 - x3, 2) + math.pow(y4 - y3, 2))
            side4 = math.sqrt(math.pow(x1 - x4, 2) + math.pow(y1 - y4, 2))
            h = min(side1 + side3, side2 + side4)/2.0#高度
            if h*dim[0] >=6:
                theta = math.atan2(y2 - y1, x2 - x1)
                top_left.append(np.array([x1 - h/2, y1 - h/2, x1 + h/2, y1 + h/2, theta, 1]))
                top_right.append(np.array([x2 - h/2, y2 - h/2, x2 + h/2, y2 + h/2, theta, 1]))
                bottom_right.append(np.array([x3 - h/2, y3 - h/2, x3 + h/2, y3 + h/2, theta, 1]))
                bottom_left.append(np.array([x4 - h/2, y4 - h/2, x4 + h/2, y4 + h/2, theta, 1]))
                ## get seg mask
                c1_x, c2_x, c3_x, c4_x, c_x = (x1 + x2)/2.0, (x2 + x3)/2.0, (x3 + x4)/2.0, (x4 + x1)/2.0, (x1 + x2 + x3 + x4)/4.0
                c1_y, c2_y, c3_y, c4_y, c_y = (y1 + y2)/2.0, (y2 + y3)/2.0, (y3 + y4)/2.0, (y4 + y1)/2.0, (y1 + y2 + y3 + y4)/4.0
                top_left_draw.polygon([x1*dim[1], y1*dim[0], c1_x*dim[1], c1_y*dim[0], c_x*dim[1], c_y*dim[0], c4_x*dim[1], c4_y*dim[0]], fill = 1)
                top_right_draw.polygon([c1_x*dim[1], c1_y*dim[0], x2*dim[1], y2*dim[0], c2_x*dim[1], c2_y*dim[0], c_x*dim[1], c_y*dim[0]], fill = 1)
                bottom_right_draw.polygon([c_x*dim[1], c_y*dim[0], c2_x*dim[1], c2_y*dim[0], x3*dim[1], y3*dim[0], c3_x*dim[1], c3_y*dim[0]], fill = 1)
                bottom_left_draw.polygon([c4_x*dim[1], c4_y*dim[0], c_x*dim[1], c_y*dim[0], c3_x*dim[1], c3_y*dim[0], x4*dim[1], y4*dim[0]], fill = 1)
        seg[0] = top_left_mask
        seg[1] = top_right_mask
        seg[2] = bottom_right_mask
        seg[3] = bottom_left_mask
        if len(top_left) == 0:
            top_left.append(np.array([-1, -1, -1, -1, 0, 0]))
            top_right.append(np.array([-1, -1, -1, -1, 0, 0]))
            bottom_right.append(np.array([-1, -1, -1, -1, 0, 0]))
            bottom_left.append(np.array([-1, -1, -1, -1, 0, 0]))

    else:
        top_left.append(np.array([-1, -1, -1, -1, 0, 0]))
        top_right.append(np.array([-1, -1, -1, -1, 0, 0]))
        bottom_right.append(np.array([-1, -1, -1, -1, 0, 0]))
        bottom_left.append(np.array([-1, -1, -1, -1, 0, 0]))

    top_left = torch.FloatTensor(np.array(top_left))
    top_right = torch.FloatTensor(np.array(top_right))
    bottom_right = torch.FloatTensor(np.array(bottom_right))
    bottom_left = torch.FloatTensor(np.array(bottom_left))

    seg = torch.from_numpy(seg).float()
    seg = seg.permute(1, 2, 0).contiguous()
    return [top_left, top_right, bottom_right, bottom_left], seg

def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri
def expansion(bbox, rate, max_shr=15):
    rate = rate * rate
    area = plg.Polygon(bbox).area()
    peri = perimeter(bbox)
    try:
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        print(offset)
        shrinked_bbox = pco.Execute(offset)
        if len(shrinked_bbox) == 0:
            return shrinked_bbox
        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            return shrinked_bbox
        #shrinked_bboxes.append(shrinked_bbox)
    except Exception as e:
        print(type(shrinked_bbox), shrinked_bbox)
        print('area:', area, 'peri:', peri)
        #shrinked_bboxes.append(bbox)
    return shrinked_bbox

def shrink(bbox, rate, max_shr=10):
    rate = rate * rate
    area = plg.Polygon(bbox).area()
    peri = perimeter(bbox)
    try:
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            return shrinked_bbox
        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            return shrinked_bbox
        #shrinked_bboxes.append(shrinked_bbox)
    except Exception as e:
        print(type(shrinked_bbox), shrinked_bbox)
        print('area:', area, 'peri:', peri)
        #shrinked_bboxes.append(bbox)

    return shrinked_bbox


def get_centerpoints(contours):
    center_points = []
    for i in range(len(contours)):
        M = cv2.moments(contours[i])  # 计算第一条轮廓的各阶矩,字典形式
        if M['m00']==0:
            return center_points
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        center_points.append([center_x, center_y])
    return center_points



def getDist_P2L(PointP, Pointa, Pointb):
    """计算点到直线的距离
        PointP：定点坐标
        Pointa：直线a点坐标
        Pointb：直线b点坐标
    """
    # 求直线方程
    A = 0
    B = 0
    C = 0
    A = Pointa[1] - Pointb[1]
    B = Pointb[0] - Pointa[0]
    C = Pointa[0] * Pointb[1] - Pointa[1] * Pointb[0]
    # 代入点到直线距离公式
    distance = 0
    distance = (A * PointP[0] + B * PointP[1] + C) / math.sqrt(A * A + B * B)

    return distance


# ***** 求两点间距离*****
def getDist_P2P(Point0, PointA):
    distance = math.pow((Point0[0] - PointA[0]), 2) + math.pow((Point0[1] - PointA[1]), 2)
    distance = math.sqrt(distance)
    return distance


def gaussian_radius(height, width, min_overlap=0.7):
  #height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)


    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)


    masked_heatmap = heatmap[y - int(top):y + int(bottom), x - int(left):x + int(right)]
    masked_gaussian = gaussian[int(radius - top):int(radius + bottom), int(radius - left):int(radius + right)]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def crop_area(im, polys, tags, crop_background=False, max_tries=5000, vis=False, img_name=None):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

    if polys.shape[0] == 0:
        return im, [], []

    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1

    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags

    for i in range(max_tries):
        # print('we have try {} times'.format(i))
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        # if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
        if xmax - xmin < 0.1 * w or ymax - ymin < 0.1 * h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []

        if len(selected_polys) == 0:
            # no text in this area
            if crop_background == True:
                im = im[ymin:ymax + 1, xmin:xmax + 1, :]
                polys = []
                tags = []

                return im, polys, tags
            else:
                continue
        else:
            if crop_background == False:
                im = im[ymin:ymax + 1, xmin:xmax + 1, :]
                polys = polys.tolist()
                polys = [polys[i] for i in selected_polys]
                polys = np.array(polys)
                polys[:, :, 0] -= xmin  # ndarray
                polys[:, :, 1] -= ymin
                polys = polys.astype(np.int32)
                polys = polys.tolist()

                tags = tags.tolist()
                tags = [tags[i] for i in selected_polys]
                return im, polys, tags
            else:
                continue
    return im, polys, tags


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    #print(y.shape)
    #print((x.shape))

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    #print(h.shape)
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    #print(masked_heatmap.shape,'**')
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
def draw_tblr(tr_mask,value,H,W):
    # dense_wh 当前多边形内部形状 要填到当前多边形内的四个方向的值
    regmap=np.zeros((H,W,4))
    index=tr_mask==1
    regmap[index]=value
    regmap=regmap.transpose(2,0,1)





    #print(index.shape,regmap.shape,regmap[index].shape)
    return regmap
    # dense_wh tr_mask geo_dist
    # dense_wh_mask =4个tr_mask


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]#4？
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    #对于heatmap不为0的区域，都会填上宽高信息
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    #print(masked_regmap.shape,masked_heatmap.shape)
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    #print(masked_regmap.shape,masked_regmap)
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap

def draw_contour_on_mask(size, cnt):
    mask = np.zeros(size, dtype='uint8')
    mask = cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask
def get_furthest_point_from_edge(size,contours):
    center_points = []
    for i in range(len(contours)):

        mask = draw_contour_on_mask((size,size), contours[i])
        d = distance_transform_edt(mask)
        cy, cx = np.unravel_index(d.argmax(), d.shape)
        center_points.append([cx,cy])
    return center_points
'''def draw_direction_score(score_map,center):'''
