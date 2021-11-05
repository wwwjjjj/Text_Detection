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
    image_show = cv2.polylines(image_show, contours, True, (0, 0, 255), 1)
    image_show = cv2.resize(image_show, (512, 512))
    if (tr is not None) and (hm is not None):
        tr =tr.astype(np.uint8)
       # hm=hm.astype(np.uint8)
        tr = cv2.cvtColor(tr * 255, cv2.COLOR_GRAY2BGR)
        tr=cv2.resize(tr,(512,512))
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
        hm = cv2.resize(hm, (512, 512))
        '''hm_t = np.ascontiguousarray(np.stack([hm_t * 255, hm_t  * 255, hm_t  * 255]).transpose(1, 2, 0))
        hm_b = np.ascontiguousarray(np.stack([hm_b , hm_b, hm_b]).transpose(1, 2, 0))
        hm_r = np.ascontiguousarray(np.stack([hm_r, hm_r , hm_r]).transpose(1, 2, 0))
        hm_l = np.ascontiguousarray(np.stack([hm_l, hm_l , hm_l]).transpose(1, 2, 0))'''


        image_show = np.concatenate([image_show, tr, hm], axis=1)#,hm_t,hm_b,hm_l,hm_r

        return image_show
    else:
        return image_show

def inference(cfg,detector, test_loader, vis_dir,output_dir,exp_name):

    total_time = 0.

    for i, ret in enumerate(test_loader):
        image = ret['input'].cuda()



        start = time.time()

        idx = 0 # test mode can only run with batch_size == 1

        # get detection result

        contours, output = detector.detect(image,ret)


        #torch.cuda.synchronize()
        end = time.time()
        total_time += end - start
        fps = (i + 1) / total_time

        # visualization
        tr_pred, hm_pred,hm_t,hm_b,hm_l,hm_r = output['tr'], output['hm'],output['hm_t'],output['hm_b'],output['hm_l'],output['hm_r']
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()

        img_show = ((img_show * cfg['train']['stds'] +cfg['train']['means']) * 255).astype(np.uint8)

        pred_vis = visualize_detection(img_show, contours, tr_pred, hm_pred,hm_t,hm_b,hm_l,hm_r)
        gt_contour = []
        for annot, n_annot in zip(ret['meta']['annotation'][idx], ret['meta']['n_annotation'][idx]):
            if n_annot.item() > 0:
                gt_contour.append(annot[:n_annot].int().cpu().numpy())



        import copy
        H, W = ret['meta']['Height'][idx].item(), ret['meta']['Width'][idx].item()
        img_show, contours = rescale_result(img_show, contours, H, W)

        gt_vis = visualize_detection(img_show, gt_contour, ret['tr'][idx][0].cpu().numpy(),
                                     ret['hm'][idx][0].cpu().numpy(), ret['hm_t'][idx][0].cpu().numpy(),
                                     ret['hm_b'][idx][0].cpu().numpy(), ret['hm_l'][idx][0].cpu().numpy(),
                                     ret['hm_r'][idx][0].cpu().numpy())
        im_vis = np.concatenate([pred_vis, gt_vis], axis=0)  # cfg['path']['vis_dir'], '{}_test'.format(opt.exp_name)
        mkdirs(vis_dir)
        path = os.path.join(cfg['path']['vis_dir'], exp_name, ret['meta']['image_id'][
            idx])  # '/home/pei_group/jupyter/Wujingjing/Pytorch_Text_Detection/output/totaltext_vis/'

        cv2.imwrite(path, im_vis)
        # write to file
        mkdirs(output_dir)

        write_to_file(contours, os.path.join(output_dir, ret['meta']['image_id'][idx].replace('jpg', 'txt')))

        print('detect {} / {} images: {}. ({:.2f} fps)'.format(i + 1, len(test_loader), ret['meta']['image_id'][idx],
                                                               fps))
    return total_time
def main(opt,cfg,test_loader):
    output_dir = os.path.join(cfg['path']['output_dir'], opt.exp_name)
    vis_dir = os.path.join(cfg['path']['vis_dir'], opt.exp_name)
    detector = TextDetector(cfg,None)

    print('Start testing Text Diffusion.')

    total_time=inference(cfg,detector, test_loader, vis_dir,output_dir,opt.exp_name)
    print("Total time of inference( area 0.15,stepsize adjustable) is : {}".format(total_time))
    # compute DetEval
    print('Computing DetEval in {}/{}'.format(cfg['path']['output_dir'], opt.exp_name))
    #output_dir = os.path.join(cfg['path']['output_dir'], opt.exp_name)
    subprocess.call(['python', '/home/pei_group/jupyter/Wujingjing/Pytorch_Text_Detection/total_text/Evaluation_Protocol/Python_scripts/Deteval.py','--tr', '0.7', '--tp', '0.6'])
    subprocess.call(['python', '/home/pei_group/jupyter/Wujingjing/Pytorch_Text_Detection/total_text/Evaluation_Protocol/Python_scripts/Deteval.py', '--tr', '0.8', '--tp', '0.4'])
    print('End.')


if __name__ == "__main__":
    # parse arguments
    # ---------------- ARGS AND CONFIGS ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Hourglass_52")
    parser.add_argument("--exp_name", type=str, default="Hourglass_test_for_diffu")
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
    if config['modelname'] == 'Hourglass':
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


        main(opt,config,test_loader)