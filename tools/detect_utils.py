import numpy as np
import cv2
import torch
import torch.nn as nn
from shapely.geometry import Polygon
import os
import math
import shapely.geometry as geometry
from scipy.spatial import Delaunay
import shapely

from shapely.ops import cascaded_union, polygonize
import errno
def _gather_feat(feat, ind, mask=None):
    print(feat.shape,ind.shape)
    dim  = feat.size(1)
    #ind  = ind.unsqueeze(2).expand(-1,  dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array([point for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        if s*(s-a)*(s-b)*(s-c)<0:
            continue
            #return None,None
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*(area+0.000001))

        # Here's the radius filter.
        #print circum_r
        if circum_r < alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    polygons=cascaded_union(triangles)
    return polygons, edge_points





def topK(heatmap,K=40,hm_thresh=0.2):
    heatmap=torch.squeeze(heatmap,dim=1)
    height,width=heatmap.shape

    topk_score,topk_ind=torch.topk(heatmap.view(1,-1),K)

    topk_ind=topk_ind[topk_score>0.2]#
    #print(topk_score)


    topk_xs = (topk_ind % width).int().float()
    topk_ys = (topk_ind / width).int().float()

    return topk_xs.tolist(),topk_ys.tolist()
def rescale_result(image, contours, H, W):
    ori_H, ori_W = image.shape[:2]
    ori_H/=4
    ori_W/=4
    image = cv2.resize(image, (W, H))
    for cont in contours:
        cont[:, 0] = (cont[:, 0] * W / ori_W).astype(int)
        cont[:, 1] = (cont[:, 1] * H / ori_H).astype(int)
    return image, contours
def mkdirs(newdir):
    """
    make directory with parent path
    :param newdir: target path
    """
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise

def approx_area_of_intersection(det_x, det_y, gt_x, gt_y):
    """
    This helper determine if both polygons are intersecting with each others with an approximation method.
    Area of intersection represented by the minimum bounding rectangular [xmin, ymin, xmax, ymax]
    """
    det_ymax = np.max(det_y)
    det_xmax = np.max(det_x)
    det_ymin = np.min(det_y)
    det_xmin = np.min(det_x)

    gt_ymax = np.max(gt_y)
    gt_xmax = np.max(gt_x)
    gt_ymin = np.min(gt_y)
    gt_xmin = np.min(gt_x)

    all_min_ymax = np.minimum(det_ymax, gt_ymax)
    all_max_ymin = np.maximum(det_ymin, gt_ymin)

    intersect_heights = np.maximum(0.0, (all_min_ymax - all_max_ymin))

    all_min_xmax = np.minimum(det_xmax, gt_xmax)
    all_max_xmin = np.maximum(det_xmin, gt_xmin)
    intersect_widths = np.maximum(0.0, (all_min_xmax - all_max_xmin))

    return intersect_heights * intersect_widths
def iou(det_x, det_y, gt_x, gt_y):
    """
    This helper determine the intersection over union of two polygons.
    """

    if approx_area_of_intersection(det_x, det_y, gt_x, gt_y) > 1: #only proceed if it passes the approximation test
        ymax = np.maximum(np.max(det_y), np.max(gt_y)) + 1
        xmax = np.maximum(np.max(det_x), np.max(gt_x)) + 1
        bin_mask = np.zeros((ymax, xmax))
        det_bin_mask = np.zeros_like(bin_mask)
        gt_bin_mask = np.zeros_like(bin_mask)

        rr, cc = polygon(det_y, det_x)
        det_bin_mask[rr, cc] = 1

        rr, cc = polygon(gt_y, gt_x)
        gt_bin_mask[rr, cc] = 1

        final_bin_mask = det_bin_mask + gt_bin_mask

        #inter_map = np.zeros_like(final_bin_mask)
        inter_map = np.where(final_bin_mask == 2, 1, 0)
        inter = np.sum(inter_map)

        #union_map = np.zeros_like(final_bin_mask)
        union_map = np.where(final_bin_mask > 0, 1, 0)
        union = np.sum(union_map)
        return inter / float(union + 1.0)
        #return np.round(inter / float(union + 1.0), 2)
    else:
        return 0


def local_max(heat,kernel=5):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
def get_center_points():

    pass
