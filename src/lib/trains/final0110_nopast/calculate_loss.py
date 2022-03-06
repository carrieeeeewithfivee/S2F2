#_backwardsmallpastmaskfutureflowloss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.losses import FocalLoss, SSIM, ForwardLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .trainer import BaseTrainer

import numpy as np
import cv2

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    warp = F.grid_sample(img, grid, mode='bilinear')

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return warp, mask.float()
    return warp

class FutureLoss(torch.nn.Module):
    def __init__(self, opt):
        super(FutureLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        
        #final1025
        self._s_det = nn.Parameter(-1.85 * torch.ones(1))
        self._s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.s_fut = nn.Parameter(-1.95 * torch.ones(1))
        self.s_flow = nn.Parameter(-1.05 * torch.ones(1))
        #self._s_id2 = nn.Parameter(-1.05 * torch.ones(1))

        #for backward warp loss, and future hm loss
        self.flow_loss_test = SSIM()
        self.future_loss = ForwardLoss()

    def forward(self, outputs, batch, future_predict=None, index_list=None, past_info = None, det=None): #index , which label to use from multiple labels
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        future_loss, p_warp_loss, fhm_loss = 0, 0, 0
        gru_id_loss = 0
        f_warp_loss = 0

        #Original loss
        index = 0
        output = outputs[0]
        #output = det
        if not opt.mse_loss:
            output['hm'] = _sigmoid(output['hm'])
        hm_loss += self.crit(output['hm'], batch['hm'][:,index]) / opt.num_stacks
        if opt.wh_weight > 0:
            wh_loss += self.crit_reg(
                output['wh'], batch['reg_mask'][:,index],
                batch['ind'][:,index], batch['wh'][:,index]) / opt.num_stacks
        if opt.reg_offset and opt.off_weight > 0:
            off_loss += self.crit_reg(output['reg'], batch['reg_mask'][:,index],
                                      batch['ind'][:,index], batch['reg'][:,index]) / opt.num_stacks
        if opt.id_weight > 0:
            id_head = _tranpose_and_gather_feat(output['id'], batch['ind'][:,index])
            id_head = id_head[batch['reg_mask'][:,index] > 0].contiguous()
            id_head = self.emb_scale * F.normalize(id_head)
            id_target = batch['ids'][:,index][batch['reg_mask'][:,index] > 0]
            id_output = self.classifier(id_head).contiguous()
            id_loss += self.IDLoss(id_output, id_target)

        #bbox detection loss
        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        #gru id loss
        '''
        #output2 = future_predict['det']
        output2 = det
        if opt.id_weight > 0:
            id_head = _tranpose_and_gather_feat(output2['id'], batch['ind'][:,index])
            id_head = id_head[batch['reg_mask'][:,index] > 0].contiguous()
            id_head = self.emb_scale * F.normalize(id_head)
            id_target = batch['ids'][:,index][batch['reg_mask'][:,index] > 0]
            id_output = self.classifier(id_head).contiguous()
            gru_id_loss += self.IDLoss(id_output, id_target)
        '''
        '''
        #Past Loss
        p_flow = future_predict['p_flows'] #flow from t-1 to t
        #backward flow, t~t-1
        warp_start = F.interpolate((past_info['input'][:,0]), size=(152,272)) #t-1
        warp_end = F.interpolate((batch['input'][:,0]), size=(152,272)) #t
        coord = p_flow.permute(0, 2, 3, 1)
        warp = bilinear_sampler(warp_start, coord, mode='bilinear', mask=False)
        if opt.mask: #mask should be on t-1 (flow from t-1 to t)
            loss_map = self.flow_loss_test(warp, warp_end) * (batch['hm'][:,index]+0.1) #label mask at t
        else:
            loss_map = self.flow_loss_test(warp, warp_end)
        p_warp_loss = p_warp_loss + (torch.flatten(loss_map, 2).mean(-1).mean()*opt.warp_weight)
        '''
        #future loss
        assert len(opt.f_weight) == opt.future_len
        for index in index_list: #0, 1, 2 #range of future len
            #future backward flow
            #'''
            f_flow = future_predict['f_flows'][index] #8, 2, 152, 272
            warp_start = F.interpolate((batch['input'][:,index+1]), size=(152,272)) #t+f
            warp_end = F.interpolate((batch['input'][:,0]), size=(152,272)) #t
            coord = f_flow.permute(0, 2, 3, 1)
            warp = bilinear_sampler(warp_start, coord, mode='bilinear', mask=False)
            if opt.mask2: #mask should be on t-1 (flow from t-1 to t)
                loss_map = self.flow_loss_test(warp, warp_end) * (batch['hm'][:,0]+0.1) #label mask at t
            else:
                loss_map = self.flow_loss_test(warp, warp_end)
            f_warp_loss = f_warp_loss + torch.flatten(loss_map, 2).mean(-1).mean()
            #'''
            #future l1 loss
            f_flows = future_predict['f_flows'][index] #8, 2, 152, 272
            _hm_loss = self.future_loss(f_flows,
                                        batch['reg_mask'][:,0],batch['ind'][:,0],batch['ids'][:,0], #forward warp loss from 0~1 0~2 0~3
                                        batch['ind'][:,index+1],batch['ids'][:,index+1])
            #add loss for each future prediction
            fhm_loss = fhm_loss + _hm_loss*opt.f_weight[index-1]

        loss = torch.exp(-self._s_det) * det_loss * 10 + torch.exp(-self._s_id) * (id_loss) + torch.exp(-self.s_fut) * (fhm_loss+f_warp_loss) + (self._s_id + self.s_fut*self._s_det)
        #loss = torch.exp(-self._s_det) * det_loss * 10 + torch.exp(-self._s_id) * (id_loss) + torch.exp(-self.s_fut) * fhm_loss + torch.exp(-self.s_flow) * p_warp_loss + (self._s_id + self.s_fut*self._s_det*self.s_flow)
        loss_stats = {'loss': loss, 'p_warp_loss': torch.zeros(1), 'f_warp_loss': f_warp_loss, 'det_loss': det_loss, 'fhm_loss': fhm_loss, 'id_loss': id_loss, 'gru_id_loss': torch.zeros(1)}
        #loss = torch.exp(-self._s_det) * det_loss * 10 + torch.exp(-self._s_id) * (id_loss) + torch.exp(-self.s_fut) * (fhm_loss) + (self._s_id + self.s_fut*self._s_det)
        #loss_stats = {'loss': loss, 'p_warp_loss': torch.zeros(1), 'f_warp_loss': torch.zeros(1), 'det_loss': det_loss, 'fhm_loss': fhm_loss, 'id_loss': id_loss, 'gru_id_loss': torch.zeros(1)}
        
        return loss, loss_stats


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)

        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.s_flow = nn.Parameter(-1.05 * torch.ones(1)) #for past flow
        #self.s_id2 = nn.Parameter(-1.05 * torch.ones(1))

        #for backward warp loss
        self.flow_loss_test = SSIM()

    def forward(self, outputs, batch, past_info = None, det=None): #index , which label to use from multiple labels
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        p_warp_loss = 0
        gru_id_loss = 0

        #calculate original loss
        index = 0
        #for s in range(opt.num_stacks):
        output = outputs[0]
        #output = det
        if not opt.mse_loss:
            output['hm'] = _sigmoid(output['hm'])
        hm_loss += self.crit(output['hm'], batch['hm'][:,index]) / opt.num_stacks
        if opt.wh_weight > 0:
            wh_loss += self.crit_reg(
                output['wh'], batch['reg_mask'][:,index],
                batch['ind'][:,index], batch['wh'][:,index]) / opt.num_stacks
        if opt.reg_offset and opt.off_weight > 0:
            off_loss += self.crit_reg(output['reg'], batch['reg_mask'][:,index],
                                      batch['ind'][:,index], batch['reg'][:,index]) / opt.num_stacks
        if opt.id_weight > 0:
            id_head = _tranpose_and_gather_feat(output['id'], batch['ind'][:,index])
            id_head = id_head[batch['reg_mask'][:,index] > 0].contiguous()
            id_head = self.emb_scale * F.normalize(id_head)
            id_target = batch['ids'][:,index][batch['reg_mask'][:,index] > 0]
            id_output = self.classifier(id_head).contiguous()
            id_loss += self.IDLoss(id_output, id_target)
        
        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        #gru id loss
        '''
        output2 = det
        if opt.id_weight > 0:
            id_head = _tranpose_and_gather_feat(output2['id'], batch['ind'][:,index])
            id_head = id_head[batch['reg_mask'][:,index] > 0].contiguous()
            id_head = self.emb_scale * F.normalize(id_head)
            id_target = batch['ids'][:,index][batch['reg_mask'][:,index] > 0]
            id_output = self.classifier(id_head).contiguous()
            gru_id_loss += self.IDLoss(id_output, id_target)
        '''
        '''
        #calculate backward loss
        if p_delta_flow is not None: 
            p_flow = p_delta_flow + coords_grid(opt.batch_size, 152, 272).to('cuda').detach()
            #backward flow
            warp_start = F.interpolate((past_info['input'][:,index]), size=(152,272)) #t-1
            warp_end = F.interpolate((batch['input'][:,index]), size=(152,272)) #t
            coord = p_flow.permute(0, 2, 3, 1)
            warp = bilinear_sampler(warp_start, coord, mode='bilinear', mask=False)
            if opt.mask:
                loss_map = self.flow_loss_test(warp, warp_end) * (batch['hm'][:,index]+0.1) #label mask is t
            else:
                loss_map = self.flow_loss_test(warp, warp_end)
            p_warp_loss = p_warp_loss + (torch.flatten(loss_map, 2).mean(-1).mean()*opt.warp_weight)
        else:
            p_warp_loss = None

        if p_warp_loss is not None:
            loss = torch.exp(-self.s_det) * det_loss * 10 + torch.exp(-self.s_id) * (id_loss) + torch.exp(-self.s_flow) * p_warp_loss + (self.s_det*self.s_flow + self.s_id) #final more id
            loss_stats = {'loss': loss, 'p_warp_loss': p_warp_loss, 'f_warp_loss': torch.zeros(1), 'det_loss': det_loss, 'fhm_loss': torch.zeros(1), 'id_loss': id_loss, 'gru_id_loss': torch.zeros(1)}
        else:
            loss = torch.exp(-self.s_det) * det_loss * 10 + torch.exp(-self.s_id) * (id_loss) + (self.s_det + self.s_id) #final more id
            #loss = torch.exp(-self.s_det) * det_loss * 10 + torch.exp(-self.s_id) * (id_loss) + (self.s_det + self.s_id) #final more id
            loss_stats = {'loss': loss, 'p_warp_loss': torch.zeros(1), 'f_warp_loss': torch.zeros(1), 'det_loss': det_loss, 'fhm_loss': torch.zeros(1), 'id_loss': id_loss, 'gru_id_loss': torch.zeros(1)}
        '''
        loss = torch.exp(-self.s_det) * det_loss * 10 + torch.exp(-self.s_id) * (id_loss) + (self.s_det + self.s_id) #final more id
        loss_stats = {'loss': loss, 'p_warp_loss': torch.zeros(1), 'f_warp_loss': torch.zeros(1), 'det_loss': det_loss, 'fhm_loss': torch.zeros(1), 'id_loss': id_loss, 'gru_id_loss': torch.zeros(1)}

        return loss, loss_stats



class MyTrainer(BaseTrainer):
    def __init__(self, opt, model, lstm_model, encoder, optimizer=None, optimizer_lstm=None):
        super(MyTrainer, self).__init__(opt, model, lstm_model, encoder, optimizer=optimizer, optimizer_lstm=optimizer_lstm)

    def _get_losses(self, opt):
        #loss_states = ['loss', 'bbox', 'id', 'fut', 'f_hm', 'p_warp', 'par_det', 'par_id', 'par_fut', 'par_flow']
        loss_states = ['loss', 'p_warp_loss', 'f_warp_loss', 'det_loss', 'fhm_loss', 'id_loss', 'gru_id_loss']
        mot_loss = MotLoss(opt)
        future_loss = FutureLoss(opt)
        return loss_states, mot_loss, future_loss

    #To Do, rewrite
    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(output['hm'], output['wh'], reg=reg,cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
