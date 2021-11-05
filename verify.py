import os
import time
import cv2
import numpy as np
import torch
import subprocess
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import yaml
import argparse
import matplotlib.pyplot as plt
from tools.augmentation import BaseTransform,Augmentation
from tools.detection import TextDetector
from os import listdir
from models.Hourglass import get_large_hourglass_net
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataset.total_text import TotalText
from tools.detect_utils import rescale_result,mkdirs
def to_cuda(cfg,*tensors):
    if len(tensors) < 2:
        return tensors[0].cuda()
    return (t.cuda() for t in tensors)

def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 1], cont[:, 0]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')
def visualize_detection(image, contours, tr=None, hm=None,hm_t=None,hm_b=None,hm_l=None,hm_r=None):
    image_show = image.copy()
    image_show = cv2.resize(image_show, (128,128))

    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    image_show = cv2.polylines(image_show, contours, True, (0, 0, 255), 3)

    if (tr is not None) and (hm is not None):
        tr =tr.astype(np.uint8)
       # hm=hm.astype(np.uint8)
        tr = cv2.cvtColor(tr * 255, cv2.COLOR_GRAY2BGR)
        '''plt.imshow(image_show)
        plt.show()
        plt.imshow(hm)
        plt.show()
        plt.imshow(hm_t)
        plt.show()
        plt.imshow(hm_b)
        plt.show()
        plt.imshow(hm_l)
        plt.show()
        plt.imshow(hm_r)
        plt.show()'''

        hm = np.stack([hm * 255, hm * 255, hm * 255]).transpose(1, 2, 0)
        hm_t = np.ascontiguousarray(np.stack([hm_t * 255, hm_t  * 255, hm_t  * 255]).transpose(1, 2, 0))
        hm_b = np.ascontiguousarray(np.stack([hm_b , hm_b, hm_b]).transpose(1, 2, 0))
        hm_r = np.ascontiguousarray(np.stack([hm_r, hm_r , hm_r]).transpose(1, 2, 0))
        hm_l = np.ascontiguousarray(np.stack([hm_l, hm_l , hm_l]).transpose(1, 2, 0))


        image_show = np.concatenate([image_show, tr, hm,hm_t,hm_b,hm_l,hm_r], axis=1)

        return image_show
    else:
        return image_show

def verify(cfg,model, test_loader, output_dir,exp_name):

    total_time = 0.
    #input_dir = "/home/pei_group/jupyter/Wujingjing/Pytorch_Text_Detection/output/totaltext_vis/Hourglass_10_26"
    #allInputs = listdir(input_dir)
    x = np.zeros((9))+0.001#linspace(0, 128, 9)
    y = np.zeros((9))
    yy=np.array([  0.28264718,  10.10464216,  18.73600682,  29.10602758,  41.2447429,
  55.56068197,  86.21147158 ,117.28852412,   0.        ])
    plt.plot(yy)
    plt.title("average error")
    plt.show()
    for i, ret in enumerate(test_loader):
        image = ret['input'].cuda()

        start = time.time()
        idx = 0 # test mode can only run with batch_size == 1

        # get model result
        output_=model(image)
        output = output_[-1]
        heat_map_t = output['hm_t'][0][0].data.cpu().numpy()
        heat_map_b = output['hm_b'][0][0].data.cpu().numpy()
        heat_map_l = output['hm_l'][0][0].data.cpu().numpy()
        heat_map_r = output['hm_r'][0][0].data.cpu().numpy()
        #print(ret['o_tr'].shape)
        gt_tr_map=ret['o_tr'][idx][0].cpu().numpy()
        gt_heat_map_t=ret['hm_t'][idx][0].cpu().numpy()
        gt_heat_map_b=ret['hm_b'][idx][0].cpu().numpy()
        gt_heat_map_l=ret['hm_l'][idx][0].cpu().numpy()
        gt_heat_map_r=ret['hm_r'][idx][0].cpu().numpy()

        error_t = np.abs(heat_map_t - gt_heat_map_t) * gt_tr_map
        error_b = np.abs(heat_map_b - gt_heat_map_b) * gt_tr_map
        error_l = np.abs(heat_map_l - gt_heat_map_l) * gt_tr_map
        error_r = np.abs(heat_map_r - gt_heat_map_r) * gt_tr_map

        for i in range(128):
            for j in range(128):
                ind=int(gt_heat_map_b[i][j]/16)
                x[ind]+=1
                y[ind]+=error_b[i][j]
                ind = int(gt_heat_map_t[i][j] / 16)
                x[ind] += 1
                y[ind] += error_t[i][j]
                ind = int(gt_heat_map_l[i][j] / 16)
                x[ind] += 1
                y[ind] += error_l[i][j]
                ind = int(gt_heat_map_r[i][j] / 16)
                x[ind] += 1
                y[ind] += error_r[i][j]
        print(y/x,x)



















def main(opt,cfg,model,test_loader):
    output_dir = os.path.join(cfg['path']['output_dir'], opt.exp_name)
    model = model.cuda()
    #cudnn.benchmark = True

    print('Start testing Text Diffusion.')

    verify(cfg,model, test_loader, output_dir,opt.exp_name)

    # compute DetEval
    print('Computing DetEval in {}/{}'.format(cfg['path']['output_dir'], opt.exp_name))
    #output_dir = os.path.join(cfg['path']['output_dir'], opt.exp_name)
    '''subprocess.call(['python', 'total_text/Evaluation_Protocol/Python_scripts/Deteval.py','--tr', '0.7', '--tp', '0.6'])
    subprocess.call(['python', 'total_text/Evaluation_Protocol/Python_scripts/Deteval.py', '--tr', '0.8', '--tp', '0.4'])
    print('End.')'''


if __name__ == "__main__":
    # parse arguments
    # ---------------- ARGS AND CONFIGS ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Hourglass")
    parser.add_argument("--exp_name", type=str, default="Hourglass_10_27")
    opt = parser.parse_args()

    # print args in train.sh
    print("--- TRAINING ARGS ---")
    print(opt)

    if not os.path.exists("configs/%s.yaml" % opt.config):
        print("*** configs/%s.yaml not found. ***" % opt.config)
        exit()

    # read yaml configs
    f = open("configs/%s.yaml" % opt.config, "r", encoding="utf-8")
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print("--- CONFIG ---")
    print(config)
    print(torch.cuda.is_available())
    print(torch.backends.cudnn.enabled)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config['modelname'] == 'Hourglass_101':
        data_test = TotalText(
            config,
            data_root=config['path']['test_data_path'],
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=512, mean=config['train']['means'],
                                    std=config['train']['stds']),
            map_transform=BaseTransform(size=config['train']['input_size'] // config['train']['downsample'],
                                        mean=config['train']['means'],
                                        std=config['train']['stds']) if config['train']['downsample'] > 0 else None,
            map_size=config['train']['input_size'] if config['train']['downsample'] == 0 else config['train'][
                                                                                                  'input_size'] //
                                                                                              config['train'][
                                                                                                  'downsample']
        )

        test_loader = data.DataLoader(
            data_test,
            batch_size=1,#config['train']['batchsize'],
            pin_memory=True,
            shuffle=False,
            drop_last=True)
        model = get_large_hourglass_net(18, config['heads'], 64,
                                        torch.load(config['path']['resume_path'])).cuda()


        main(opt,config,model,test_loader)