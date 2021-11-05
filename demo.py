import sys
import tensorwatch as tw
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import pandas as pd
import numpy.random as random
import torch
import torch.nn
import Polygon as plg
import argparse

from models.Hourglass import get_large_hourglass_net
from tools.detection import TextDetector

from tools.augmentation import Normalize,Resize,Compose,BaseTransform
from tools.utils import find_bottom,find_long_edges,split_edge_seqence
import os
from tools.detect_utils import local_max,mkdirs
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.io as io
import cv2
import yaml
import copy
from tools.utils import shrink,get_centerpoints,draw_umich_gaussian,draw_dense_reg
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataset.total_text import TextInstance
class TextInstance(object):

    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text

        remove_points = []

        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)



    def find_centerline(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text

        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        self.center=[center_points[int(len(center_points) / 2)][0],center_points[int(len(center_points) / 2)][1]]


        return center_points


    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)
def judge(x,L):
    if x>=0 and x<L:
        return True
    return False

def parse_mat(mat_path):
    annotation=io.loadmat(mat_path)
    polygons=[]
    for cell in annotation['polygt']:
        if len(cell)<=3 or len(cell[1][0])<=3:
            continue
        x=cell[1][0]
        y=cell[3][0]

        if len(cell) >= 5 and len(cell[4]) > 0:
            text = cell[4][0]
        else:
            text = ''
        try:
            ori=cell[5][0]
        except:
            ori='c'
        points=np.stack([x,y]).T.astype(np.int32)
        polygons.append(TextInstance(points,ori,text))

    return polygons
def generate_label(polygons):
    max_annotation=128
    H = 128
    W = 128
    #1.1得到文本区域的0 1 编码
    tr_mask = np.zeros((H,W), np.uint8)
    #1.2得到文本区域的mask，如果文本是未表明文字的，就mask掉
    '''shrink_polygons=[]
        for i in range(len(polygons)):
            shrink_polygons.append(shrink(polygons[i].points,0.4))'''
    shrink_polygons = []
    for i in range(len(polygons)):

        shrink_polygons.append(TextInstance(shrink(polygons[i].points, 0.1), polygons[i].orient, polygons[i].text))

    train_mask=np.ones((H,W),np.uint8)
    i=0
    for polygon in shrink_polygons:
        if len(polygon.points.astype(np.int32))==0:
            cv2.fillPoly(tr_mask,[polygons[i].points.astype(np.int32)],color=1)
        else:
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], color=1)
        if polygon.text=='#':
            cv2.fillPoly(train_mask, [polygons[i].points.astype(np.int32)], color=0)
        i=i+1
    ori_tr_mask=np.zeros((H,W), np.uint8)
    for polygon in polygons:
        temp = np.zeros((H, W), np.uint8)
        cv2.fillPoly(temp, [polygon.points.astype(np.int32)], color=255)
        plt.imshow(temp)
        plt.show()
        cv2.fillPoly(ori_tr_mask,[polygon.points.astype(np.int32)],color=1)


    #2.得到文本中心点
    #6.得到文本中心点offsets
    center_points = []
    index_of_ct=np.zeros((max_annotation), dtype=np.int64)
    offsets_mask=np.zeros((max_annotation), dtype=np.uint8)
    center_offsets=np.zeros((max_annotation,2), dtype=np.float32)
    for i in range(len(polygons)):

        center_x = polygons[i].center[0]
        center_y = polygons[i].center[1]
        center_points.append([int(center_x), int(center_y)])
        offsets_mask[i] = 1
        index_of_ct[i]=int(center_y)*W+int(center_x)
        if index_of_ct[i]<0 or index_of_ct[i]>=H*W or judge(center_x,H)==False or judge(center_y,W)==False:
            #print("error")
            index_of_ct[i]=0
            offsets_mask[i]=0

        center_offsets[i][0] = int(center_x) -center_x
        center_offsets[i][1] = int(center_y) -center_y


    #3.得到四个方向的位置编码
    geo_map = np.zeros((4, H,W),np.float32)
    #4.得到四个方向的最大距离
    dense_wh = np.zeros((4,H,W))
    geo_max_dis=np.zeros((max_annotation,4),dtype=np.float32)
    #每个在文本区间中的像素点都可以预测wh
    #print(dense_wh_mask.shape)
    for k in range(len(polygons)):
        m = np.zeros((4, H, W), np.float32)
        score_map=np.zeros((H,W),np.int32)
        cv2.fillPoly(score_map, [polygons[k].points.astype(np.int32)], color=(1, 0))
        mmax = np.array([1e-6, 1e-6, 1e-6, 1e-6])
        for i in range(H):
            for j in range(W):
                dist = cv2.pointPolygonTest(polygons[k].points.astype(int), [i, j], False)
                if dist < 0:
                    continue

                 # 上 x=x y=y-1 左
                x = i
                y = j
                while (y>=0 and score_map[y][x]):
                    y = y - 1
                m[0][j][i] = np.abs(y - j)
                if mmax[0] < m[0][j][i]:
                    mmax[0] = m[0][j][i]
                # 下 x=x y=y+1 右
                x = i
                y = j
                while (y<W and score_map[y][x]):
                    y = y + 1
                m[1][j][i] = np.abs(y - j)
                if mmax[1] < m[1][j][i]:
                    mmax[1] = m[1][j][i]
                # 左 x=x-1 y=y 上
                x = i
                y = j
                while (x>=0 and score_map[y][x]):
                    x = x - 1
                m[2][j][i] = np.abs(x - i)
                if mmax[2] < m[2][j][i]:
                    mmax[2] = m[2][j][i]
                   # 右 x=x+1 y=y 下
                x = i
                y = j
                while (x<H and score_map[y][x]):
                    x = x + 1
                m[3][j][i] = np.abs(x - i)
                if mmax[3] < m[3][j][i]:
                    mmax[3] = m[3][j][i]

        for i in range(H):
            for j in range(W):
                dist = cv2.pointPolygonTest(polygons[k].points.astype(int), [i, j], False)
                if dist < 0:
                    continue

                    #m[0][j][i] = float(m[0][j][i]) / mmax[0]
                    #m[1][j][i] = float(m[1][j][i]) / mmax[1]
                    #m[2][j][i] = float(m[2][j][i]) / mmax[2]
                    #m[3][j][i] = float(m[3][j][i]) / mmax[3]
                for tt in range(4):#min(m[tt][j][i],geo_map[tt][j][i])
                    temp=0 if  m[tt][j][i]>0 and geo_map[tt][j][i]>0 else m[tt][j][i]
                    geo_map[tt][j][i]=temp
        geo_max_dis[k]=mmax
        #对于在任意一个多边形内的点，dense_wh被填上其所在的多边形的四个方向的最大值，如果同时在
        #两个多边形内部，那么填上较小的那个多边形对应的4个值
        #5.得到中心点heatmap
        #print(geo_max_dis)
    heatmap = np.zeros((H, W), np.float32)
    for k in range(len(polygons)):
        rect = cv2.minAreaRect(polygons[k].points.astype(np.int32))
        box = cv2.boxPoints(rect)
        area = plg.Polygon(box).area()
        #radius=int(np.sqrt(area))
        radius = int(min(np.sqrt(area),np.abs(cv2.pointPolygonTest(polygons[k].points.astype(np.int32), center_points[k], True))))+1
        draw_umich_gaussian(heatmap, center_points[k], radius)
        # dense_wh tr_mask geo_dist
        # dense_wh_mask =4个tr_mask
        temp_mask=np.zeros((H,W),dtype=np.uint8)
        cv2.fillPoly(temp_mask,[polygons[k].points.astype(np.int32)],1)
    ret = {'hm': heatmap[np.newaxis,np.newaxis, :], 'trm': train_mask[np.newaxis,np.newaxis, :], 'tr': tr_mask[np.newaxis,np.newaxis, :],
           'o_tr': ori_tr_mask[np.newaxis,np.newaxis, :],
           'hm_t': (geo_map[0])[np.newaxis,np.newaxis, :], 'hm_b': (geo_map[1])[np.newaxis,np.newaxis, :], 'hm_l': (geo_map[2])[np.newaxis,np.newaxis, :],
           'hm_r': (geo_map[3])[np.newaxis,np.newaxis, :],
           'offsets': center_offsets, 'off_mask': offsets_mask,
           'center_points': index_of_ct
           }
    return ret
def visualize_detection(image, contours, tr=None, hm=None,hm_t=None,hm_b=None,hm_l=None,hm_r=None):
    image_show = image.copy()
    image_show = cv2.resize(image_show, (128,128))

    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    image_show = cv2.polylines(image_show, contours, True, (0, 0, 255), 1)
    image_show = cv2.resize(image_show, (512, 512))
    if (tr is not None) and (hm is not None):
        tr =tr.astype(np.uint8)
       # hm=hm.astype(np.uint8)
        tr = cv2.cvtColor(tr * 255, cv2.COLOR_GRAY2BGR)
        tr=cv2.resize(tr,(512,512))


        hm = np.stack([hm * 255, hm * 255, hm * 255]).transpose(1, 2, 0)
        hm = cv2.resize(hm, (512, 512))
        '''hm_t = np.ascontiguousarray(np.stack([hm_t * 255, hm_t  * 255, hm_t  * 255]).transpose(1, 2, 0))
        hm_b = np.ascontiguousarray(np.stack([hm_b , hm_b, hm_b]).transpose(1, 2, 0))
        hm_r = np.ascontiguousarray(np.stack([hm_r, hm_r , hm_r]).transpose(1, 2, 0))
        hm_l = np.ascontiguousarray(np.stack([hm_l, hm_l , hm_l]).transpose(1, 2, 0))'''


        image_show = np.concatenate([image_show, tr, hm], axis=1)#,hm_t,hm_b,hm_l,hm_r

        return image_show
    else:
        return image_show

def demo_only_diffu(opt,cfg):
    image = Image.open(opt.data_dir)
    image_ = np.array(image)
    plt.imshow(cv2.resize(image_, (128, 128)))
    plt.show()
    transform = Compose([
        Resize(cfg['train']['input_size']),
        Normalize(cfg['train']['means'], cfg['train']['stds'])
    ])
    polygons = parse_mat(opt.anno_dir)
    for i in range(len(polygons)):
        polygons[i].find_centerline(15)
    image, polygons = transform(copy.copy(image_), copy.copy(polygons))





    resize=Resize(config['train']['input_size']// config['train']['downsample'])
    print(image.shape)
    _,polygons=resize(copy.copy(image),polygons)
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis, :]
    input_image = torch.from_numpy(image).cuda()
    detector = TextDetector(cfg, None)
    ret=generate_label(polygons)
    gt_map = {'hm': torch.from_numpy(ret['hm']).cuda(),  # 'tr_m':ret['tr_m'].cuda(),
              'o_tr': torch.from_numpy(ret['o_tr']).cuda(),
              'hm_t': torch.from_numpy(ret['hm_t']).cuda(), 'hm_b': torch.from_numpy(ret['hm_b']).cuda(),
              'hm_l': torch.from_numpy(ret['hm_l']).cuda(), 'hm_r': torch.from_numpy(ret['hm_r']).cuda(),
              'trm': torch.from_numpy(ret['trm']).cuda(), 'tr': torch.from_numpy(ret['tr']).cuda(),
              # 'dense_wh': ret['dense_wh'].cuda( ), 'dense_wh_mask': ret['dense_wh_mask'].cuda( ),
              'center_points': torch.from_numpy(ret['center_points']).cuda()}

    contours, output = detector.detect(input_image,gt_map)

    tr_pred, hm_pred, hm_t, hm_b, hm_l, hm_r = output['tr'], output['hm'], output['hm_t'], output['hm_b'], output[
        'hm_l'], output['hm_r']
    plt.imshow(ret['o_tr'][0][0])
    plt.show()
    plt.imshow(tr_pred)
    plt.show()
    plt.imshow(hm_pred)
    plt.show()
    plt.imshow(hm_t)
    plt.show()
    plt.imshow(hm_b)
    plt.show()
    plt.imshow(hm_r)
    plt.show()
    plt.imshow(hm_l)
    plt.show()
    img_show = input_image[0].permute(1, 2, 0).cpu().numpy()
    img_show = ((img_show * cfg['train']['stds'] + cfg['train']['means']) * 255).astype(np.uint8)
    print(img_show.shape)
    pred_vis = visualize_detection(img_show, contours, tr_pred, hm_pred)
    plt.imshow(pred_vis)
    plt.show()
    #mkdirs(os.path.join(opt['demo_dir']))
    #print(opt['demo_dir'])
    path = os.path.join(opt.demo_dir,
                        'demo.jpg')  # '/home/pei_group/jupyter/Wujingjing/Pytorch_Text_Detection/output/totaltext_vis/'
    cv2.imwrite(path, pred_vis)

def demo(opt,cfg,model):
    image = Image.open(opt.data_dir)
    image_ = np.array(image)
    plt.imshow(cv2.resize(image_, (128, 128)))
    plt.show()
    transform = Compose([
        Resize(cfg['train']['input_size']),
        Normalize(cfg['train']['means'], cfg['train']['stds'])
    ])

    image, _ = transform(image_)
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis, :]
    input_image = torch.from_numpy(image).cuda()

    model.eval()
    detector = TextDetector(cfg, model)
    contours, output = detector.detect(input_image,None)
    tr_pred, hm_pred, hm_t, hm_b, hm_l, hm_r = output['tr'], output['hm'], output['hm_t'], output['hm_b'], output[
        'hm_l'], output['hm_r']
    plt.imshow(tr_pred)
    plt.show()
    plt.imshow(hm_pred)
    plt.show()
    plt.imshow(hm_t)
    plt.show()
    plt.imshow(hm_b)
    plt.show()
    plt.imshow(hm_r)
    plt.show()
    plt.imshow(hm_l)
    plt.show()
    img_show = input_image[0].permute(1, 2, 0).cpu().numpy()

    img_show = ((img_show * cfg['train']['stds'] + cfg['train']['means']) * 255).astype(np.uint8)

    pred_vis = visualize_detection(img_show, contours, tr_pred, hm_pred, hm_t, hm_b, hm_l, hm_r)
    gt_contour = []

    polygons = parse_mat(opt.anno_dir)
    for i in range(len(polygons)):
        polygons[i].find_centerline(15)


    gt_transform =BaseTransform(size=config['train']['input_size'] // config['train']['downsample'],
                                        mean=config['train']['means'],
                                        std=config['train']['stds'])if config['train']['downsample'] > 0 else None
    _, polygons = gt_transform(copy.copy(img_show), polygons)

    points = np.zeros((cfg['max_annotation'], cfg['max_points'], 2))
    length = np.zeros(cfg['max_annotation'], dtype=int)

    for i, polygon in enumerate(polygons):
        pts = polygon.points
        points[i, :pts.shape[0]] = polygon.points
        length[i] = pts.shape[0]

    for annot, n_annot in zip(points,length):
        if n_annot.item() > 0:
            gt_contour.append(annot[:n_annot])
    print(pred_vis.shape)
    mkdirs(opt['demo_dir'])
    print(opt['demo_dir'])
    path = os.path.join(opt['demo_dir'],'demo.jpg')  # '/home/pei_group/jupyter/Wujingjing/Pytorch_Text_Detection/output/totaltext_vis/'

    cv2.imwrite(path, pred_vis)
    plt.imshow(pred_vis)
    plt.show()









if __name__=='__main__':
    # ---------------- ARGS AND CONFIGS ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Hourglass_52")
    parser.add_argument('--modelfile', type=str,
                        default="/home/pei_group/jupyter/Wujingjing/Text_Detection/save_models/Hourglass_101_shrink_best.pth")#1000
    parser.add_argument('--data_dir', type=str, default="/home/pei_group/jupyter/Wujingjing/data/totaltext/Images/Test/img633.jpg")
    parser.add_argument('--anno_dir', type=str,
                        default="/home/pei_group/jupyter/Wujingjing/data/totaltext/gt/Test/poly_gt_img633.mat")
    parser.add_argument('--demo_dir', type=str,
                        default="/home/pei_group/jupyter/Wujingjing/Pytorch_Text_Detection/demo_result/")

    opt = parser.parse_args()
    print("--- TRAINING ARGS ---")
    print(opt)
    f = open("configs/%s.yaml" % opt.config, "r", encoding="utf-8")
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print("--- CONFIG ---")
    print(config)
    model = get_large_hourglass_net(18, config['heads'], 64,
                                    torch.load(config['path']['resume_path'])).cuda()

    dummy_input = torch.rand(8, 3, 512, 512).cuda()

    #demo(opt,config,model)
    demo_only_diffu(opt, config)
