import numpy as np
from torch.utils.data import Dataset
import os
import Polygon as plg
from tools.utils import judge, pil_load_img,find_bottom,draw_tblr,find_long_edges,draw_dense_reg,split_edge_seqence,norm2, shrink,get_centerpoints,draw_umich_gaussian,get_furthest_point_from_edge
import cv2
import scipy.io as io

import copy
import matplotlib.pyplot as plt
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

class TotalText(Dataset):
    def __init__(self,cfg, data_root, ignore_list=None, is_training=True, transform=None,map_transform=None,map_size=512):
        super().__init__()
        self.data_root = data_root
        self.is_training = is_training
        self.cfg=cfg
        self.transform=transform
        self.map_transform=map_transform
        self.map_size=map_size

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))

        self.annotation_list = ['poly_gt_{}.mat'.format(img_name.replace('.jpg', '').lstrip('0')) for img_name in
                                self.image_list]




    def generate_label(self,image,polygons):
        max_annotation=128
        H = self.map_size
        W = self.map_size
        #1.1得到文本区域的0 1 编码
        tr_mask = np.zeros((H,W), np.uint8)
        #1.2得到文本区域的mask，如果文本是未表明文字的，就mask掉
        '''shrink_polygons=[]
        for i in range(len(polygons)):
            shrink_polygons.append(shrink(polygons[i].points,0.4))'''
        shrink_polygons = []
        for i in range(len(polygons)):

            shrink_polygons.append(TextInstance(shrink(polygons[i].points, 0.5), polygons[i].orient, polygons[i].text))

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
                    for tt in range(4):
                        temp=min(m[tt][j][i],geo_map[tt][j][i]) if  m[tt][j][i]>0 and geo_map[tt][j][i]>0 else m[tt][j][i]
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
            dense_wh=dense_wh*(dense_wh!=0)+draw_tblr(temp_mask,geo_max_dis[k],H,W)*(dense_wh==0)
            #draw_dense_reg(dense_wh, heatmap, center_points[k], geo_max_dis[k],radius)
            #draw_dense_reg(dense_wh_mask,heatmap,center_points[k],[1,1,1,1],radius)
            #print(geo_max_dis[k][0])
            '''cv2.fillPoly(dense_wh[0], [polygons[k].points.astype(np.int32)], int(geo_max_dis[k][0]))
            cv2.fillPoly(dense_wh[1], [polygons[k].points.astype(np.int32)], int(geo_max_dis[k][1]))
            cv2.fillPoly(dense_wh[2], [polygons[k].points.astype(np.int32)], int(geo_max_dis[k][2]))
            cv2.fillPoly(dense_wh[3], [polygons[k].points.astype(np.int32)], int(geo_max_dis[k][3]))'''
        '''print(np.nonzero(dense_wh.reshape(-1)))
        print(np.nonzero(dense_wh_mask.reshape(-1)))'''
        #print(dense_wh[dense_wh_mask==1])
        #print(np.nonzero(dense_wh_mask.reshape(-1)==1))
        dense_wh_mask = np.stack([tr_mask, tr_mask, tr_mask,
                                  tr_mask])  # np.zeros((4, H,W),np.uint8)#np.stack([train_mask,train_mask,train_mask,train_mask]).astype(np.uint8)

        '''plt.imshow(dense_wh[0])
        plt.show()
        plt.imshow(tr_mask)
        plt.show()
        plt.imshow(dense_wh_mask[0])
        plt.show()'''
        '''dense_wh,dense_wh_mask,'''
        return tr_mask,ori_tr_mask,train_mask,geo_map,index_of_ct,heatmap,center_offsets,offsets_mask

    def parse_mat(self,mat_path):
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


    def __getitem__(self, item):
        #得到编码后的四个方向的map，中心点（heatmap），每个像素点和真实中心点之间的偏移量，每个像素点所在的文本区域的大概宽高
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)
        #print(image_id)

        # Read image data
        image = pil_load_img(image_path)

        H,W,_=image.shape

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)

        polygons=self.parse_mat(annotation_path)

        '''if self.is_training:
            polygons=shrink_polygons'''
        '''for i in range(len(shrink_polygons)):
            shrink_polygons[i].find_centerline(15)'''
        for i in range(len(polygons)):
            polygons[i].find_centerline(15)

        if self.transform:
            image, polygons = self.transform(copy.copy(image), copy.copy(polygons))

        if self.map_transform:
            _,polygons=self.map_transform(copy.copy(image),copy.copy(polygons))
            #_, shrink_polygons = self.map_transform(copy.copy(image), copy.copy(shrink_polygons))
        points = np.zeros((self.cfg['max_annotation'], self.cfg['max_points'], 2))
        length = np.zeros(self.cfg['max_annotation'], dtype=int)

        for i, polygon in enumerate(polygons):
            pts = polygon.points
            points[i, :pts.shape[0]] = polygon.points
            length[i] = pts.shape[0]

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'annotation': points,
            'n_annotation': length,
            'Height': H,
            'Width': W
        }
        tr_mask,ori_tr_mask,train_mask,m,index_of_ct,heatmap,center_offsets,offsets_mask=self.generate_label(image,polygons)


        image = image.transpose(2,0,1)  # 0, 3, 1, 2
        ret={'input':image,'hm':heatmap[np.newaxis,:],'trm':train_mask[np.newaxis,:],'tr':tr_mask[np.newaxis,:],'o_tr':ori_tr_mask[np.newaxis,:],
             'hm_t':(m[0])[np.newaxis,:],'hm_b':(m[1])[np.newaxis,:],'hm_l':(m[2])[np.newaxis,:],'hm_r':(m[3])[np.newaxis,:],
             'offsets':center_offsets,'off_mask':offsets_mask,
             'center_points':index_of_ct,'meta':meta
             }


        return ret#image,tr_mask[np.newaxis,:],m,meta,shrink_mask[np.newaxis,:],heatmap[np.newaxis,:]



    def __len__(self):
        return len(self.image_list)
