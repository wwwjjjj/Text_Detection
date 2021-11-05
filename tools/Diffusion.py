import matplotlib.pyplot as plt
import numpy as np
import cv2
import alphashape
from tools.detect_utils import alpha_shape as alpha
from descartes import PolygonPatch
import shapely
class Diffusion():
#漫步
    def __init__(self,x,y, m0,m1,m2,m3,num_points=5000):
        self.num_points = num_points
        # 所有随机漫步都始于(center)
        self.x_values = [x]#list(center_points[:, 0])
        self.y_values = [y]#list(center_points[:, 1])

        self.pre_points=[[x,y]]
        self.pre_choose=[-1,-1,-1,-1]
        self.pre_area=0
        self.hull=None
        self.walk_flag=True


        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3


    def plot_polygon(self,polygon):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        margin = .3

        x_min, y_min, x_max, y_max = polygon.bounds

        ax.set_xlim([x_min - margin, x_max + margin])
        ax.set_ylim([y_min - margin, y_max + margin])
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
        ax.invert_yaxis()
        patch = PolygonPatch(polygon, fc='#ffdede', ec='#ffdede', fill=True, zorder=-1)
        plt.show()

        return fig, patch
    def judge(self,x,y):
        flag=(self.m0[y][x]<0) or (self.m1[y][x]<0) or (self.m2[y][x]<0) or (self.m3[y][x]<0)
        return flag
    def fill_walk(self,flag):
        """计算随机漫步包含的所有点"""
        if self.walk_flag==False:
            return len(self.x_values)


        #print(np.array(self.pre_points,np.float32))
        hull = cv2.convexHull(np.array(self.pre_points,np.float32), clockwise=True, returnPoints=True)
        area=cv2.contourArea(hull)

        if flag==1 or self.pre_area!=0 and np.abs(area-self.pre_area)<0.2*area:#self.pre_

            if len(np.array(self.pre_points))<=3:
                alpha_shape=None
            else:
                alpha_shape,_=alpha(np.array(self.pre_points),8)

            i = 10
            if alpha_shape is not None and isinstance(alpha_shape, shapely.geometry.polygon.Polygon) is False:
                while isinstance(alpha_shape, shapely.geometry.polygon.Polygon) is False:
                    i=i+4
                    alpha_shape,_=alpha(np.array(self.pre_points),i)
            '''while len(np.array(alpha_shape.__geo_interface__['coordinates'][-1]))>=50:
                i=i+10
                alpha_shape, _ = alpha(np.array(self.pre_points), i)'''

            self.hull=None if alpha_shape is None else np.array(alpha_shape.__geo_interface__['coordinates'][-1])
            '''points_2d=[(point[0],point[1]) for point in self.pre_points]
            fig, ax = plt.subplots()
            ax.scatter(*zip(*points_2d))
            plt.show()
            ax.add_patch(PolygonPatch(alpha_shape, alpha=10))'''

            '''self.plot_polygon(alpha_shape)'''
            self.walk_flag=False
        else:
            self.pre_area=area
        destinations = [ [0, -1], [0, 1],
                        [-1, 0], [1, 0]]
        step = 10
        # print(self.x_values[-1])
        next_ax = []
        next_ay = []
        next_points=[]

        #print(len(self.pre_x))
        for i in range(len(self.pre_points)):

            value_x = self.pre_points[i][0]
            value_y = self.pre_points[i][1]
            if self.judge(int(value_y),int(value_x)) is True:
               continue

            for choose in range(4):
                if choose==0:
                    score=self.m0
                    #step=self.step_size[0]*128
                if choose==1:
                    score=self.m1
                    #step = self.step_size[1]*128
                if choose==2:
                    score=self.m2
                   # step = self.step_size[2]*128
                if choose==3:
                    score=self.m3
                   # step = self.step_size[3]*128
                x_step=score[int(value_y)][int(value_x)] * destinations[choose][0]
                y_step=score[int(value_y)][int(value_x)] * destinations[choose][1]

                deltax = max(0.5,min(0.7,(1-np.abs(x_step/128))))* x_step#* step
                deltay = max(0.5,min(0.7,(1-np.abs(y_step/128))))* y_step# * step
                deltax=0.7*x_step
                deltay=0.7*y_step


                next_x = value_x + deltax
                next_y = value_y + deltay

                if next_x<0 or next_x>=128 or next_y<0 or next_y>=128 or score[int(next_y)][int(next_x)]>=score[int(value_y)][int(value_x)] score[]:
                    continue
                self.pre_choose[choose] = score[int(value_y)][int(value_x)]
                next_ax.append(next_x)
                next_ay.append(next_y)
                next_points.append([next_x,next_y])



        self.pre_points.clear()
        self.pre_points.extend(next_points)
        #print(len(self.pre_x))
        self.x_values.extend(next_ax)

        self.y_values.extend(next_ay)
        return len(self.x_values)
