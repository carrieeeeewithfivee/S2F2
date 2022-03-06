#fairmot
import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import cv2
import torch.nn.functional as F

from models.model import create_model, load_model, create_decoder, create_encoder
from models.decode import mot_decode
from tracking_utils.utils import *
from tracking_utils.log import logger
from tracking_utils.kalman_filter import KalmanFilter
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState
from utils.post_process import ctdet_post_process, f_ctdet_post_process, bbox_post_process
from utils.image import get_affine_transform
from models.utils import _tranpose_and_gather_feat
import copy
torch.autograd.set_detect_anomaly(True)

#for vis
import ntpath
from colorhash import ColorHash
from .flow_vis import flow_to_color, find_rad_minmax
import datetime

#test
from scipy import spatial

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


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30, future_bbox=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.f_b = future_bbox #tlbr, [f, 4]

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.f_b = new_track.f_b

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class Tracker(object):
    def __init__(self, opt, frame_rate=30, name='test'):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        #create and load model
        print('Creating model...')
        model = create_model(opt.arch, opt.heads, opt.head_conv)
        decoder = create_decoder(opt.future_len, opt.decoder_arch)
        encoder = create_encoder(opt.encoder_arch)
        self.model, _, self.decoder, _, _, self.encoder = load_model(model, opt.load_model, lstm_model=decoder, encoder=encoder)
        self.decoder = self.decoder.to(opt.device)
        self.model = self.model.to(opt.device)
        self.encoder = self.encoder.to(opt.device)
        self.decoder.eval()
        self.model.eval()
        self.encoder.eval()

        #for tracks
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.kalman_filter = KalmanFilter()

        #for warmup stuff
        self.warmup_len = opt.warmup_len
        self.step_size = opt.step_size
        self.warmup_index = 0
        #the s=10 encoder states
        self.state_queue = [torch.cuda.FloatTensor(1, 128, 152, 272).fill_(0) for i in range (self.step_size)]

        #inference
        self.poly = opt.polygon
        self.flow = opt.save_flow
        basename = opt.load_model.split('/')[-2]
        self.name = "_".join([basename, name])

        #try to improve tracking with forecasting results
        self.f_centers = []

    def post_process(self, dets, meta): #resize predictions to original size
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def f_post_process(self, f_bboxes, meta):
        processed = []
        for _bbox in f_bboxes:
            _bbox = _bbox.detach().cpu().numpy()
            _bbox = _bbox.reshape(1, -1, _bbox.shape[2]) #1, 500, 4
            _bbox = bbox_post_process(_bbox.copy(), [meta['c']], [meta['s']],meta['out_height'], meta['out_width'], self.opt.num_classes)
            processed.append(_bbox[0])
        return processed

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0, timer=None):
        #for tracking
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        #for visulization
        if self.flow:
            grid0 = torch.meshgrid(torch.arange(152), torch.arange(272))
            grid0 = torch.stack(grid0[::-1], dim=0).float().permute(1,2,0).numpy()
            blk = img0.copy()
            img1 = img0.copy()
            rawimg = img0.copy()

        #for resizing
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        
        run_future = True
        if self.warmup_index < self.warmup_len*self.step_size+1: #after 30 frames
            run_future = False

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            #run detection
            outputs, features = self.model(im_blob)
            output = outputs[-1]
            RunLSTM = False

            if self.opt.lstm:
                #calculate encoder index
                encoder_index = self.frame_id%self.step_size
                if self.frame_id == 0: #first one
                    state = self.encoder(self.state_queue[encoder_index],features,torch.ones(1))
                else:
                    state = self.encoder(self.state_queue[encoder_index],features,torch.zeros(1))
                self.state_queue[encoder_index] = state
                #run LSTM
                if run_future: #after 30 frames
                    #run decoder
                    future_predict = self.decoder(state) #p_delta_flow is t~t-1 flow
                    RunLSTM = True
                else:
                    self.warmup_index = self.warmup_index + 1

            #output of bbox model
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            _id_feature = output['id']
            id_feature = F.normalize(_id_feature, dim=1)

            if RunLSTM:
                #"""
                if self.flow:
                    '''vis future flow'''
                    for i in [2]: #[2,1,0]
                        warp_coords = future_predict['f_flows'][i]
                        warp_coords = warp_coords.permute(0,2,3,1)
                        flow = warp_coords.squeeze(0).cpu().numpy()-grid0
                        if i == 2:
                            rad_max, _min, _max = find_rad_minmax(flow)
                        flow_color = flow_to_color(flow, convert_to_bgr=False, clip_flow=[_min, _max] ,rad_max=rad_max)
                        path = '../demos/'+self.name+'_flow'+str(i)+'/'
                        os.makedirs(path, exist_ok=True)
                        cv2.imwrite(path+str(self.frame_id)+'.jpg', flow_color)
                #"""
                future_xy = future_predict['f_flows'] #3 tensors of [1, 2, 152, 272]
            else:
                future_xy = None
                
            #process ids
            dets, inds, f_bbox = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K, future_xy=future_xy)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        #process detections original
        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1] #500, 5, x1 y1 x2 y2 conf
        remain_inds = dets[:, 4] > self.opt.conf_thres
        
        #if det close to any future of before
        #"""
        if len(self.f_centers) == 10: #the 0'th one is t-1, will have t+1 pred
            ret = np.asarray(self.f_centers[0][:,0,:]).copy()
            self.f_centers = self.f_centers[1:]
            f_center = (ret[:,:2]+ret[:,2:])/2
            searchtree = spatial.KDTree(f_center) #center of f bboxes
            ret2 = np.asarray(dets).copy()
            t_center = (ret2[:,:2]+ret2[:,2:4])/2
            for i in range(500):
                if searchtree.query(t_center[i])[0] < 10:
                    if ret2[i, 4] > self.opt.conf_thres/2:
                        remain_inds[i] = True
        #"""
        #remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds] #n, 4
        id_feature = id_feature[remain_inds]
        if f_bbox:
            f_bbox = [_bbox[:, remain_inds, :] for _bbox in f_bbox] #1 n 4
            f_bbox = self.f_post_process(f_bbox, meta) #future, batch, n, 4
            if f_bbox:
                f_bbox = np.array(f_bbox).transpose(1, 0, 2) #f, n, 4 -> n,f,4
            if len(dets) > 0:
                self.f_centers.append(f_bbox)
                detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30, f_b) for (tlbrs, f, f_b) in zip(dets[:, :5], id_feature, f_bbox)]
            else:
                detections = []
        elif len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        #output_stracks = self.globaltrack.update(detections)
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        #print(len(self.tracked_stracks))
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        """
        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        """

        '''check vis'''
        #"""
        if self.flow:
            for track in output_stracks:
                tlwh = track.tlwh
                f_box = track.f_b
                tid = track.track_id
                obj_id = int(tid)
                x1, y1, w, h = tlwh
                color = get_color(abs(tid))
                if self.poly:
                    #draw polygon
                    contours = [[x1, y1+h]]
                    if f_box is not None:
                        for box in f_box:
                            contours.append([box[0],box[3]])
                        for box in reversed(f_box):
                            contours.append([box[2],box[3]])
                    contours.append([x1+w, y1+h])
                    contours = np.array(contours, dtype=np.int32)
                    cv2.fillPoly(blk, pts = [contours], color=color)
                    #img0 = cv2.addWeighted(img0, 0.7, blk, 0.3, 0)
                    img1 = cv2.addWeighted(blk, 0.05, img1, 0.95, 0)
                    intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                    cv2.rectangle(img1, intbox[0:2], intbox[2:4], color, 4)
                
                #draw rectangle
                if f_box is not None:
                    for box in f_box:
                        intbox = tuple(map(int, (box[0], box[1], box[2], box[3])))
                        cv2.rectangle(img0, intbox[0:2], intbox[2:4], color, 2)
                intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                cv2.rectangle(img0, intbox[0:2], intbox[2:4], color, 2)

            #save
            path = '../demos/'+self.name+'_box/'
            os.makedirs(path, exist_ok=True)
            cv2.imwrite(path+str(self.frame_id)+'.jpg', img0)
            
            if self.poly:
                path = '../demos/'+self.name+'_poly/'
                os.makedirs(path, exist_ok=True)
                cv2.imwrite(path+str(self.frame_id)+'.jpg', img1)

            path = '../demos/'+self.name+'_raw/'
            os.makedirs(path, exist_ok=True)
            cv2.imwrite(path+str(self.frame_id)+'.jpg', rawimg)
        #"""
        return output_stracks #, img0

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb