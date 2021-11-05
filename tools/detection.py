import numpy as np
import cv2
import torch
from tools.detect_utils import local_max,topK
from tools.Diffusion import Diffusion

import matplotlib.pyplot as plt
def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

class TextDetector(object):

    def __init__(self, cfg,model):
        self.model = model
        self.tr_thresh = cfg['test']['tr_thresh']
        self.hm_thresh = cfg['test']['hm_thresh']
        self.cfg=cfg

        # evaluation mode
        if model is not None:
            self.model.eval()



    def detect_contours(self,image, tr_pred, heat_map,hm_t,hm_b,hm_l,hm_r):
# * tr_pred
        top_x, top_y = topK((heat_map[0][0]), self.cfg['max_annotation'],self.cfg['test']['hm_thresh'])
        #print(top_x)
        '''plt.imshow((heat_map[0][0]*tr_pred>0.2).cpu().detach().numpy())
        plt.show()'''

        #conts, _ = cv2.findContours((heat_map[0][0] * tr_pred), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(top_x,top_y)
        #conts,_=cv2.findContours(np.array(tr_pred.cpu(),dtype=np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #不好，因为还是有一些文本区域粘连了
        num=len(top_x)

        '''print(len(conts))
        flag = {i: 1 for i in range(len(conts))}
        for i in range(len(conts)):
            for j in range(len(top_x)):
                if cv2.pointPolygonTest(conts[i],[top_x[j],top_y[j]],False)>=0:
                    if flag[i]>=0:
                        flag[i] = j if (heat_map[0][0] * tr_pred)[int(top_x[j])][int(top_y[j])]>(heat_map[0][0] * tr_pred)[int(top_x[flag[i]])][int(top_y[flag[i]])] else flag[i]
                    else:
                        flag[i] = j'''
        instances = []
        for i in range(num):
            #if flag[i]==-1:
            #    continue
            #instances.append((Diffusion(int(top_x[flag[i]]), int(top_y[flag[i]]), hm_t, hm_b, hm_l, hm_r)))
            instances.append((Diffusion(int(top_x[i]), int(top_y[i]), hm_t, hm_b, hm_l, hm_r)))

        i = 0
        while i < self.cfg['test']['max_diffusion']:
            i = i + 1
            print(i)
            flag=0
            if i==self.cfg['test']['max_diffusion']:
                flag=1
            points_x = []
            points_y = []
            nums = []
            stop_count = 0
            for j in range(len(instances)):
                if instances[j].walk_flag == False:
                    stop_count += 1
                    points_x.extend(instances[j].x_values)
                    points_y.extend(instances[j].y_values)
                    continue

                nums.append(instances[j].fill_walk(flag))
                points_x.extend(instances[j].x_values)
                points_y.extend(instances[j].y_values)



            if stop_count == len(instances):
                break
            plt.xlim((0, 128))
            plt.ylim((0, 128))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(i)

            plt.scatter(points_x, points_y, cmap='Blues', edgecolor='none')
            # plt.scatter(148, 176, c='green', edgecolors='none', s=100)
            ax = plt.gca()
            ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
            ax.invert_yaxis()
            plt.show()

        polygons = []
        map_result = np.zeros((128, 128), np.uint8)
        for i in range(len(instances)):
            if instances[i].hull is None:
                continue
            polygons.append(np.array(instances[i].hull.squeeze()).astype(np.int32))
            cv2.fillPoly(map_result, [np.array(instances[i].hull.squeeze()).astype(np.int32)], color=255)
        #print(len(polygons))
        return polygons,map_result#self.postprocessing(image, detect_result, tr_pred_mask)

    def detect(self, image,ret):
        """
        :param image:
        :return:
        """


        if ret is None:
            # get model output

            # inference需要text region, heat map，tblr map
            output_ = self.model(image)
            image = image[0].data.cpu().numpy()
            output = output_[-1]
            tr_pred = output['tr'][0].softmax(dim=0)[1] > self.tr_thresh  # .data.cpu().numpy()[1]
            # heat_map=output['hm'][0][0].data.cpu().numpy()
            heat_map = local_max(_sigmoid(output['hm']))
            heat_map_t = output['hm_t'][0][0].data.cpu().numpy()
            heat_map_b = output['hm_b'][0][0].data.cpu().numpy()
            heat_map_l = output['hm_l'][0][0].data.cpu().numpy()
            heat_map_r = output['hm_r'][0][0].data.cpu().numpy()
        else:
            tr_pred = ret['tr'][0][0]#.data.cpu().numpy()# .data.cpu().numpy()[1]
            # heat_map=output['hm'][0][0].data.cpu().numpy()
            heat_map = local_max(ret['hm'])#.data.cpu().numpy()

            heat_map_t = ret['hm_t'][0][0].data.cpu().numpy()
            heat_map_b = ret['hm_b'][0][0].data.cpu().numpy()
            heat_map_l = ret['hm_l'][0][0].data.cpu().numpy()
            heat_map_r = ret['hm_r'][0][0].data.cpu().numpy()


        # find text contours
        #print(heat_map.shape)
        contours,map_result = self.detect_contours(image, tr_pred, heat_map,heat_map_t,heat_map_b,heat_map_l,heat_map_r)  # (n_tcl, 3)

        output = {
            'image': image,
            'tr': tr_pred.data.cpu().numpy(),
            'hm': _sigmoid(output['hm']).data.cpu().numpy()[0][0] if ret is None else ret['hm'][0][0].cpu().numpy(),
            'hm_t':heat_map_t,
            'hm_b':heat_map_b,
            'hm_l':heat_map_l,
            'hm_r':heat_map_r,
            'map_result':map_result

        }
        return contours, output

