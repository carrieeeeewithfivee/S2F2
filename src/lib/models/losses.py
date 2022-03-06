# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat
import torch.nn.functional as F
from .ssim import ssim
#for testing
from utils.image import gaussian_radius, draw_umich_gaussian
import numpy as np
import math

#from https://github.com/lliuz/ARFlow/blob/e92a8bbe66f0ced244267f43e3e55ad0fe46ff3e/losses/flow_loss.py#L3
class SMOOTH(nn.Module):
  def __init__(self):
    super(SMOOTH, self).__init__()
    self.func_smooth = smooth_grad_2nd

  def forward(self, flow, im1_scaled):
    loss = []
    loss += [self.func_smooth(flow, im1_scaled, 10)]
    return sum([l.mean() for l in loss])

def smooth_grad_2nd(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy

#from pytorch_msssim import ssim, SSIM
class SSIM(nn.Module):
  '''nn.Module warpper for SSIM loss'''
  def __init__(self):
    super(SSIM, self).__init__()
    self._ssim = _ssim

  def forward(self, out, target):
    return self._ssim(out, target)

def _ssim(pred, gt):
  ssim_loss_map = ssim(pred, gt, data_range=1, size_average=False, sim_map=True) # return a map #[8, 3, 142, 262]) 152, 272
  dw = int((272 - 262) / 2)  # width padding
  dh = int((152 - 142) / 2)  # height padding
  ssim_loss_map = 1.0-F.pad(ssim_loss_map, (dh, dh, dw, dw), "constant", 1)
  return ssim_loss_map

class ForwardWarp(nn.Module):
  '''
  nn.Module warpper for ForwardWarp loss (not used in final version)
  get future hm center position, create new heatmap, return heatmap
  to do: predict wh 
  '''
  def __init__(self):
    super(ForwardWarp, self).__init__()
    self.gaussian_radius = _gaussian_radius
    self.draw_umich_gaussian = _draw_umich_gaussian

  def forward(self, flow, mask, index, wh):
    pred = _tranpose_and_gather_feat(flow, index) #
    mask_2 = mask.unsqueeze(2).expand_as(pred).float()
    mask_4 = mask.unsqueeze(2).expand_as(wh).float()
    wh = wh * mask_4
    xy = pred * mask_2

    #create new hm #b, c, h, w
    hm = torch.cuda.FloatTensor(flow.shape[0], 1, flow.shape[2], flow.shape[3]).fill_(0)
    for b in range(flow.shape[0]): #batch
      temp = 0
      for _wh, _xy in zip(wh[b,:,:], xy[b,:,:]):
        w = _wh[0]+_wh[2]
        h = _wh[1]+_wh[3]
        x, y = _xy
        if h > 0 and w > 0 and x > 0 and y > 0 and x<152 and y<272:
          """
          test_h = h.clone().cpu().detach().numpy()
          test_w = w.clone().cpu().detach().numpy()
          test_x = x.clone().cpu().detach().numpy()
          test_y = y.clone().cpu().detach().numpy()
          test_radius = self.test_gaussian_radius((math.ceil(h), math.ceil(w)))
          ct = np.array([test_x, test_y], dtype=np.float32)
          ct_int = ct.astype(np.int32)
          self.test_draw_umich_gaussian(test_hm[b, 0], ct_int, test_radius) #not self.opt.mse_loss
          """
          radius = self.gaussian_radius((torch.ceil(h), torch.ceil(w)))
          radius = torch.clamp(radius, min=0)
          self.draw_umich_gaussian(hm[b, 0], [x,y], radius) #not self.opt.mse_loss
        else:
          temp = temp + 1
          
      #change back to tensor for test
      #test_hm = torch.from_numpy(test_hm).to('cuda')
    return hm

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y = torch.arange(-m,m+1).cuda()
    x = torch.transpose(torch.unsqueeze(torch.arange(-n,n+1),0) ,0, 1).cuda()
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < 2e-15] = 0

    return h

def _draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  x, y = center
  x = int(x)
  y = int(y)
  radius = int(radius)
  height, width = heatmap.shape[0:2]
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  #masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(heatmap[y - top:y + bottom, x - left:x + right].shape) > 0: # TODO debug
    heatmap[y - top:y + bottom, x - left:x + right] = torch.max(heatmap[y - top:y + bottom, x - left:x + right], masked_gaussian * k)
  else:
    print("Not Enter!")
  return heatmap

def _gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size
  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  r = torch.stack([r1,r2,r3],0)
  return torch.min(r)

""" numpy for testing
def test_gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    print("$$$$$$$", m, n)
    y, x = np.ogrid[-m:m+1,-n:n+1]
    #print("!!!!!!!!!!!",y, x)
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    #print("$$$$$$$$$", h)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    #print("!!!!!!!!!$$$$$$$$$", h)
    return h

def _test_draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = test_gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])
  radius = int(radius)
  height, width = heatmap.shape[0:2]
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  print("check1 : ",np.sum(masked_heatmap))
  print("check2 : ",np.sum(heatmap))
  return heatmap

def _test_gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size
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
"""

class ForwardLoss(nn.Module):
  '''
  nn.Module warpper for ForwardLoss loss
  get future hm center position, calculate loss
  '''
  def __init__(self):
    super(ForwardLoss, self).__init__()

  def forward(self, flow, mask, index, ids, index2, ids2):

    pred = _tranpose_and_gather_feat(flow, index)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    xy = pred * mask

    #reorder
    reordered_ind2 = torch.cuda.FloatTensor(flow.shape[0],500,2).fill_(0)

    mask_index = torch.cuda.FloatTensor(flow.shape[0],500,1).fill_(0)
    #add 1/12
    #reordered_ind2 = xy.clone().detach()
    for b in range(flow.shape[0]):
      order = []
      for i in ids2[b]:
        if i==0:
          order.append(-1)
        else:
          try:
            order.append(int((ids[b] == i).nonzero(as_tuple=True)[0]))
          except:
            order.append(-1)
      #order = [int(torch.flatten((ids[b] == i).nonzero(as_tuple=False))[0]) if i!=0 else -1 for i in ids2[b]]
      for i, x in enumerate(order):
        if x != -1:
          reordered_ind2[b][x][0] = index2[b][i]%272 #ToDo: debug!
          reordered_ind2[b][x][1] = index2[b][i]/272
          mask_index[b][x][0] = 1
        #else: #add 1/12 not work too slow
        #  xy[b][x][0] = 0
        #  xy[b][x][1] = 0
          #not calculate those xy points that don't have future?
    xy = xy * mask_index
    loss = F.l1_loss(xy, reordered_ind2, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class ForwardWhLoss(nn.Module):
  '''
  nn.Module warpper for ForwardWhLoss loss
  get future hm center position, calculate loss, calculate wh loss
  '''
  def __init__(self):
    super(ForwardWhLoss, self).__init__()

  def forward(self, flow, p_wh, mask, index, ids, wh, index2, ids2, wh2):
    #print('flow: ', flow.shape) #6, 2, 152, 272]
    #print('mask: ', mask.shape) #6, 500

    #coords
    pred = _tranpose_and_gather_feat(flow, index)
    mask2 = mask.unsqueeze(2).expand_as(pred).float()
    xy = pred * mask2

    #whs
    p_wh = _tranpose_and_gather_feat(p_wh, index)
    mask4 = mask.unsqueeze(2).expand_as(p_wh).float()
    pred_wh = p_wh * mask4

    #reorder
    reordered_ind2 = torch.cuda.FloatTensor(flow.shape[0],500,2).fill_(0)
    reordered_wh2 = torch.cuda.FloatTensor(flow.shape[0],500,4).fill_(0)
    zero = torch.cuda.FloatTensor(1).fill_(0)
    for b in range(flow.shape[0]):
      order = []
      for i in ids2[b]:
        if i==0:
          order.append(-1)
        else:
          try:
            order.append(int((ids[b] == i).nonzero(as_tuple=True)[0]))
          except:
            order.append(-1)
      #order = [int(torch.flatten((ids[b] == i).nonzero(as_tuple=False))[0]) if i!=0 else -1 for i in ids2[b]]
      for i, x in enumerate(order):
        if x != -1:
          reordered_ind2[b][x][0] = index2[b][i]%272
          reordered_ind2[b][x][1] = index2[b][i]/272
          reordered_wh2[b][x] = wh2[b][i]
    
    #calculate delta wh
    delta_wh = reordered_wh2 - wh
    delta_wh = torch.where(delta_wh != -wh, delta_wh, zero) * mask4 #remove 1 exist, 2 is zero, make bigger

    #index loss
    loss = F.l1_loss(xy, reordered_ind2, size_average=False)
    loss = loss / (mask2.sum() + 1e-4)
    #delta wh loss
    wh_loss = F.l1_loss(pred_wh, delta_wh, size_average=False)
    #wh_loss = F.l1_loss(pred_wh, reordered_wh2 * 10, size_average=False)
    wh_loss = wh_loss / (mask4.sum() + 1e-4)

    return loss, wh_loss

def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''

  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pred = torch.clamp(pred, 1e-9, 1-1e-9)
  #print(torch.log(pred))
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds #yuwen
  #pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds #carful if pred = 0 will get -inf
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss
