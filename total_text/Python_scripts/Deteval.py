# modified from https://github.com/cs-chan/Total-Text-Dataset/blob/master/Evaluation_Protocol/Python_scripts/Deteval.py
import numpy as np

from os import listdir
from scipy import io
from total_text.Python_scripts.polygon_wrapper import iod
from total_text.Python_scripts.polygon_wrapper import area_of_intersection
from total_text.Python_scripts.polygon_wrapper import area

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

# basic opts#"E:/Python Project/Pytorch_Text_Detection/output/totaltext_text/Hourglass_10_23"
parser.add_argument('--input_dir', default="/home/pei_group/jupyter/Wujingjing/Pytorch_Text_Detection/output/totaltext_text/Hourglass_test_for_diffu", help='Model output directory')
parser.add_argument('--tr', type=float, default=0.7, help='Recall threshold')
parser.add_argument('--tp', type=float, default=0.6, help='Precision threshold')
args = parser.parse_args()

"""
Input format: y0,x0, ..... yn,xn. Each detection is separated by the end of line token ('\n')'
"""

input_dir =args.input_dir# 'output/{}'.format(args.exp_name)
gt_dir = '/home/pei_group/jupyter/Wujingjing/data/totaltext/gt/Test'#'E:/Python Project/data/totaltext/gt/Test'
fid_path = 'Python_Pascal_result_last_check.txt'

allInputs = listdir(input_dir)


def input_reading_mod(input_dir, input):
    """This helper reads input from txt files"""
    with open('%s/%s' % (input_dir, input), 'r') as input_fid:
        pred = input_fid.readlines()
    det = [x.strip('\n') for x in pred]
    return det


def gt_reading_mod(gt_dir, gt_id):
    """This helper reads groundtruths from mat files"""
    gt_id = gt_id.split('.')[0]
    gt = io.loadmat('%s/poly_gt_%s.mat' % (gt_dir, gt_id))
    gt = gt['polygt']
    return gt


def detection_filtering(detections, groundtruths, threshold=0.5):
    for gt_id, gt in enumerate(groundtruths):
        if (gt[5] == '#') and (gt[1].shape[1] > 1):
            gt_x = list(map(int, np.squeeze(gt[1])))
            gt_y = list(map(int, np.squeeze(gt[3])))
            for det_id, detection in enumerate(detections):
                detection = detection.split(',')
                detection = list(map(int, detection[0:-1]))
                det_y = detection[0::2]
                det_x = detection[1::2]

                det_gt_iou = iod(det_x, det_y, gt_x, gt_y)
                if det_gt_iou > threshold:
                    detections[det_id] = []

            detections[:] = [item for item in detections if item != []]
    return detections

def sigma_calculation(det_x, det_y, gt_x, gt_y):
    """
    sigma = inter_area / gt_area
    """
    #print(area(gt_x, gt_y),area(det_x, det_y))
    if area(gt_x, gt_y)==0:
        return 0
    return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) / area(gt_x, gt_y)), 2)

def tau_calculation(det_x, det_y, gt_x, gt_y):
    """
    tau = inter_area / det_area
    """
    #print(area(gt_x, gt_y), area(det_x, det_y))
    if area(det_x, det_y)==0:
        return 0
    return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) / area(det_x, det_y)), 2)




##############################Initialization###################################
global_tp = 0
global_fp = 0
global_fn = 0
global_sigma = []
global_tau = []
tr = args.tr
tp = args.tp
fsc_k = 0.8
k = 2
###############################################################################

for i, input_id in enumerate(tqdm(allInputs)):

    if (input_id != '.DS_Store'):
        detections = input_reading_mod(input_dir, input_id)
        groundtruths = gt_reading_mod(gt_dir, input_id)
        detections = detection_filtering(detections, groundtruths)  # filters detections overlapping with DC area
        dc_id = np.where(groundtruths[:, 5] == '#')
        groundtruths = np.delete(groundtruths, (dc_id), (0))

        local_sigma_table = np.zeros((groundtruths.shape[0], len(detections)))
        local_tau_table = np.zeros((groundtruths.shape[0], len(detections)))
        for gt_id, gt in enumerate(groundtruths):
            if len(detections) > 0:
                for det_id, detection in enumerate(detections):
                    detection = detection.split(',')
                    detection = list(map(int, detection[:-2]))
                    det_y = detection[0::2]
                    det_x = detection[1::2]
                    gt_x = list(map(int, np.squeeze(gt[1])))
                    gt_y = list(map(int, np.squeeze(gt[3])))

                    local_sigma_table[gt_id, det_id] = sigma_calculation(det_x, det_y, gt_x, gt_y)

                    local_tau_table[gt_id, det_id] = tau_calculation(det_x, det_y, gt_x, gt_y)

        global_sigma.append(local_sigma_table)
        global_tau.append(local_tau_table)

global_accumulative_recall = 0
global_accumulative_precision = 0
total_num_gt = 0
total_num_det = 0

def one_to_one(local_sigma_table, local_tau_table, local_accumulative_recall,
               local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
               gt_flag, det_flag):

    for gt_id in range(num_gt):# 对于ground truth内的每个元素
        qualified_sigma_candidates = np.where(local_sigma_table[gt_id, :] > tr)#当前gt和det的交集占gt的值超过tr
        num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]
        qualified_tau_candidates = np.where(local_tau_table[gt_id, :] > tp)
        num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]#当前gt和det的交集占det的值超过tp


        if (num_qualified_sigma_candidates == 1) and (num_qualified_tau_candidates == 1):
        #当前gt和各个det的交集占gt大于某个阈值，且占det大于某个阈值，而且这种det仅有一个，说明这俩是一一对应的关系
            global_accumulative_recall = global_accumulative_recall + 1.0
            global_accumulative_precision = global_accumulative_precision + 1.0
            local_accumulative_recall = local_accumulative_recall + 1.0
            local_accumulative_precision = local_accumulative_precision + 1.0

            gt_flag[0, gt_id] = 1
            matched_det_id = np.where(local_sigma_table[gt_id, :] > tr)
            det_flag[0, matched_det_id] = 1
    return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag

def one_to_many(local_sigma_table, local_tau_table, local_accumulative_recall,
               local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
               gt_flag, det_flag):
    for gt_id in range(num_gt):
        #skip the following if the groundtruth was matched
        if gt_flag[0, gt_id] > 0:
            continue

        non_zero_in_sigma = np.where(local_sigma_table[gt_id, :] > 0)#当前gt和det的交集占gt的值大于0
        num_non_zero_in_sigma = non_zero_in_sigma[0].shape[0]
        #如果存在多个det，其和当前gt的交集占gt的值大于阈值
        if num_non_zero_in_sigma >= k:
            ####search for all detections that overlaps with this groundtruth
            qualified_tau_candidates = np.where((local_tau_table[gt_id, :] >= tp) & (det_flag[0, :] == 0))
            num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]
            # 如果只有一个det，其和gt的交集占gt的值大于tp，问题继续转化为1对1问题
            if num_qualified_tau_candidates == 1:
                if ((local_tau_table[gt_id, qualified_tau_candidates] >= tp) and (local_sigma_table[gt_id, qualified_tau_candidates] >= tr)):
                    #became an one-to-one case

                    global_accumulative_recall = global_accumulative_recall + 1.0
                    global_accumulative_precision = global_accumulative_precision + 1.0
                    local_accumulative_recall = local_accumulative_recall + 1.0
                    local_accumulative_precision = local_accumulative_precision + 1.0

                    gt_flag[0, gt_id] = 1
                    det_flag[0, qualified_tau_candidates] = 1
            elif (np.sum(local_sigma_table[gt_id, qualified_tau_candidates]) >= tr):
            #如果所有和gt有交集的det和gt交集占gt的和超过了tr
                gt_flag[0, gt_id] = 1
                det_flag[0, qualified_tau_candidates] = 1

                global_accumulative_recall = global_accumulative_recall + fsc_k
                global_accumulative_precision = global_accumulative_precision + num_qualified_tau_candidates * fsc_k

                local_accumulative_recall = local_accumulative_recall + fsc_k
                local_accumulative_precision = local_accumulative_precision + num_qualified_tau_candidates * fsc_k

    return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag

def many_to_many(local_sigma_table, local_tau_table, local_accumulative_recall,
               local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
               gt_flag, det_flag):
    for det_id in range(num_det):
        #对于det
        # skip the following if the detection was matched
        if det_flag[0, det_id] > 0:
            continue

        non_zero_in_tau = np.where(local_tau_table[:, det_id] > 0)
        num_non_zero_in_tau = non_zero_in_tau[0].shape[0]#与det交集不为0的gt

        if num_non_zero_in_tau >= k:#如果有多个和当前det交集不为0的gt
            ####search for all detections that overlaps with this groundtruth
            qualified_sigma_candidates = np.where((local_sigma_table[:, det_id] >= tp) & (gt_flag[0, :] == 0))
            #寻找该det是否有和某些gt的交集占某些gt的比值大于tp
            num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]

            if num_qualified_sigma_candidates == 1:
                #如果该det真的有和某个gt的交集占该gt的比值大于tp
                if ((local_tau_table[qualified_sigma_candidates, det_id] >= tp) and (local_sigma_table[qualified_sigma_candidates, det_id] >= tr)):
                    #became an one-to-one case
                    global_accumulative_recall = global_accumulative_recall + 1.0
                    global_accumulative_precision = global_accumulative_precision + 1.0
                    local_accumulative_recall = local_accumulative_recall + 1.0
                    local_accumulative_precision = local_accumulative_precision + 1.0

                    gt_flag[0, qualified_sigma_candidates] = 1
                    det_flag[0, det_id] = 1
            elif (np.sum(local_tau_table[qualified_sigma_candidates, det_id]) >= tp):
                det_flag[0, det_id] = 1
                gt_flag[0, qualified_sigma_candidates] = 1

                global_accumulative_recall = global_accumulative_recall + num_qualified_sigma_candidates * fsc_k
                global_accumulative_precision = global_accumulative_precision + fsc_k

                local_accumulative_recall = local_accumulative_recall + num_qualified_sigma_candidates * fsc_k
                local_accumulative_precision = local_accumulative_precision + fsc_k
    return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag

fid = open(fid_path, 'w')
l_one_recall = 0
l_one_precision = 0
l_many_recall = 0
l_many_precision = 0
l_many_many_recall = 0
l_many_many_precision = 0
for idx in range(len(global_sigma)):

    local_sigma_table = global_sigma[idx]
    local_tau_table = global_tau[idx]

    num_gt = local_sigma_table.shape[0]
    num_det = local_sigma_table.shape[1]

    total_num_gt = total_num_gt + num_gt
    total_num_det = total_num_det + num_det

    local_accumulative_recall = 0
    local_accumulative_precision = 0
    gt_flag = np.zeros((1, num_gt))
    det_flag = np.zeros((1, num_det))


    #######first check for one-to-one case##########

    local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
    gt_flag, det_flag = one_to_one(local_sigma_table, local_tau_table,
                                  local_accumulative_recall, local_accumulative_precision,
                                  global_accumulative_recall, global_accumulative_precision,
                                  gt_flag, det_flag)

    l_one_recall+=local_accumulative_recall
    l_one_precision+=local_accumulative_precision
    local_accumulative_recall_=local_accumulative_recall
    local_accumulative_precision_=local_accumulative_precision
    #######then check for one-to-many case##########
    local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
    gt_flag, det_flag = one_to_many(local_sigma_table, local_tau_table,
                                   local_accumulative_recall, local_accumulative_precision,
                                   global_accumulative_recall, global_accumulative_precision,
                                   gt_flag, det_flag)
    l_many_recall+=local_accumulative_recall-local_accumulative_recall_
    l_many_precision+=local_accumulative_precision-local_accumulative_precision_
    local_accumulative_recall_=local_accumulative_recall
    local_accumulative_precision_=local_accumulative_precision
    #######then check for many-to-many case##########
    local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
    gt_flag, det_flag = many_to_many(local_sigma_table, local_tau_table,
                                    local_accumulative_recall, local_accumulative_precision,
                                    global_accumulative_recall, global_accumulative_precision,
                                    gt_flag, det_flag)
    l_many_many_recall+=local_accumulative_recall-local_accumulative_recall_
    l_many_many_precision+=local_accumulative_precision-local_accumulative_precision_

    try:
        local_precision = local_accumulative_precision / num_det
    except ZeroDivisionError:
        local_precision = 0

    try:
        local_recall = local_accumulative_recall / num_gt
    except ZeroDivisionError:
        local_recall = 0

    str_write = ('%s: Precision = %.4f - Recall = %.4f\n' % (allInputs[idx], local_precision, local_recall))
    fid.write(str_write)
fid.close()

try:
    recall = global_accumulative_recall / total_num_gt
except ZeroDivisionError:
    recall = 0

try:
    precision = global_accumulative_precision / total_num_det
except ZeroDivisionError:
    precision = 0

try:
    f_score = 2*precision*recall/(precision+recall)
except ZeroDivisionError:
    f_score = 0

fid = open(fid_path, 'a')
str_write = ('Precision = %.4f - Recall = %.4f - Fscore = %.4f\n' % (precision, recall, f_score))
fid.write(str_write)
fid.close()

print('Input: {}'.format(input_dir))
print('Config: tr: {} - tp: {}'.format(tr, tp))
print(str_write)
print('Done.')
print(total_num_gt,total_num_det)
print(global_accumulative_recall,global_accumulative_precision)
print(l_one_recall,l_one_precision)
print(l_many_recall,l_many_precision)
print(l_many_many_recall,l_many_many_precision)