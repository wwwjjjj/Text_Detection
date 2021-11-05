import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import SmoothL1Loss
from torch.autograd import Variable
import numpy as np

def _neg_loss(pred, gt):

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    num_neg  = neg_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    #print(num_pos,pos_loss.item(),neg_loss.item())
    '''if num_pos!=0:
        loss=loss-pos_loss/num_pos
    if num_neg!=0:
        loss=loss-neg_loss/num_neg
    '''
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)
def _gather_feat(feat, ind, mask=None):

    dim  = feat.size(2)#c
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)#bs,128,c

    torch.gather(feat, 1,ind)

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat[:,:128,:]

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()#bs,h,w,c
    feat = feat.view(feat.size(0), -1, feat.size(3))#bs,w*h,c
    feat = _gather_feat(feat, ind)

    return feat

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        #print(pred*mask,target*mask)

        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y


class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit =FocalLoss()# torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        #self.crit_=F.smooth_l1_loss()#SmoothL1Loss()#FocalLoss()
        self.opt = opt
        self.parts = ['t', 'l', 'b', 'r']
        #self.crit_wh = torch.nn.L1Loss(reduction='sum')
        self.crit_reg = RegL1Loss()
    def ohem(self, predict, target, train_mask, negative_ratio=3.):

        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()

        n_pos = pos.float().sum()

        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = 0.
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = 100
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()
    def forward(self,output,gt_map):
        pred_map={}
        opt = self.opt
        train_mask = gt_map['trm'].view(-1).bool()
        hm_loss, geo_loss, tr_loss,off_loss = 0, 0, 0,0
        for i in range(opt['num_stack']):
            output_=output[i]
            pred_hm = _sigmoid(output_['hm'])
            if i == opt['num_stack']-1:
                pred_map['hm']=pred_hm
            # 1.计算heat map的损失
            hm_loss += self.crit(pred_hm, gt_map['hm'])/opt['num_stack']


            # 2.计算四个方向的map损失

            for p in self.parts:
                tag = 'hm_{}'.format(p)

                # pred_hm_t=_sigmoid(output_[tag])v
                if i==opt['num_stack']-1:
                    pred_map[tag] = output_[tag]
                pred_hm_t = output_[tag].contiguous().view(-1)
                gt_hm_t = gt_map[tag].contiguous().view(-1)
                geo_loss += F.smooth_l1_loss(pred_hm_t[train_mask], gt_hm_t[train_mask])/opt['num_stack']

            # 3.计算宽高损失 去掉
            # mask_weight = gt_map['dense_wh_mask'].sum() + 1e-4

            '''wh_loss+=self.crit_wh(output_['dense_wh']*gt_map['dense_wh_mask'],
                                       gt_map['dense_wh']*gt_map['dense_wh_mask']#wh_pred[gt_map['dense_wh_mask']],gt_wh_map[gt_map['dense_wh_mask']]#* gt_map['dense_wh_mask'],
                                 ) /mask_weight'''

            # dense_mask=(gt_map['dense_wh_mask']==1)
            # wh_pred=output_['dense_wh'][dense_mask]
            # print(gt_map['dense_wh'].shape)

            # wh_gt=gt_map['dense_wh'][dense_mask]
            # ones = wh_gt.new(wh_pred.size()).fill_(1.).double()
            # wh_loss=F.smooth_l1_loss(wh_pred / wh_gt, ones)
            # print(wh_gt)
            # print(wh_pred)
            # print(wh_loss.item())

            # 4.计算offsets 损失
            off_map = _sigmoid(output_['offsets'])

            off_loss += self.crit_reg(off_map, gt_map['off_mask'],
                                      gt_map['center_points'], gt_map['offsets'])/opt['num_stack']

            # 5.计算tr损失

            tr_mask = output_['tr'].permute(0, 2, 3, 1).contiguous().view(-1, 2)
            tr_loss += self.ohem(tr_mask, gt_map['tr'].contiguous().view(-1).long(), train_mask.view(-1).long())/opt['num_stack']
            if i == opt['num_stack']-1:
                pred_map['tr'] = output_['tr'][:, 1, :, :]

        loss = opt['train']['hm_weight'] * hm_loss+ opt['train']['geo_weight'] * geo_loss+opt['train']['off_weight'] *off_loss+opt['train']['tr_weight']*tr_loss#
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'geo_loss': geo_loss,'off_loss':off_loss,'tr_loss':tr_loss}#'wh_loss':wh_loss,

        return loss, loss_stats,pred_map

