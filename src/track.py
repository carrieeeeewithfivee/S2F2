from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import time

from tracker.tracker import Tracker

from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.sequence_loader as datasets
import pandas as pd
import pickle

from tracking_utils.utils import mkdir_if_missing
from opts import opts
import re

#To do, rewrite
def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        save_format = '{frame},{seqid},{id},{x1},{y1},{w},{h},1\n'
        #save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n' #for test

    if data_type == 'mot':
        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids in results:
                if data_type == 'kitti':
                    frame_id -= 1
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    f.write(line)
    elif data_type == 'kitti':
        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids in results:
                if data_type == 'kitti':
                    frame_id -= 1
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    f.write(line)
    else:
        with open(filename, 'w') as f: #has future
            for frame_id, tlwhs, track_ids, seqid in results:
                for tlwh, track_id, seq in zip(tlwhs, track_ids, seqid):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    #for test
                    """
                    if seq != 0:
                        continue
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    """

                    line = save_format.format(frame=frame_id, seqid=seq, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = Tracker(opt, frame_rate=frame_rate) #, name=seq)
    timer = Timer()
    results = []
    all_time = 0
    times = 0
    
    for i, (path, img, img0) in enumerate(dataloader):
        if opt.vis_limit:
            if i > 100:
                break
        if isinstance(path, str):
            frame_id = int(path.split('/')[-1].replace('.jpg','')) #use file name
        else:
            frame_id = path

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        torch.cuda.synchronize()
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0, timer)
        torch.cuda.synchronize()
        timer.toc()
        
        #save future
        online_tlwhs = []
        online_ids = []
        online_seqids = []
            
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            f_box = t.f_b
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_seqids.append(0)
            #future stuff
            if f_box is not None:
                for idx, box in enumerate(f_box):
                    vertical = (box[2]-box[0]) / (box[3]-box[1]) > 1.6
                    if (box[2]-box[0]) * (box[3]-box[1]) > opt.min_box_area and not vertical:
                        online_tlwhs.append([box[0],box[1],box[2]-box[0],box[3]-box[1]])
                        online_ids.append(tid)
                        online_seqids.append(idx+1)
        # save results
        results.append((frame_id, online_tlwhs, online_ids, online_seqids))
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='val_test',
         save_images=False, save_videos=False):
    exp_name = opt.exp_name
    logger.setLevel(logging.INFO)
    result_root = os.path.join('..', 'validates', exp_name, 'results')
    mkdir_if_missing(result_root)
    #data_type = 'mot'
    data_type = opt.task

    # run tracking
    accs = []
    f_accs = []
    #data_dicts = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join('..', 'validates', exp_name, 'outputs', seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        
        if opt.run_model:
            dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
            meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
            frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
            nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename, save_dir=output_dir, frame_rate=frame_rate, seq=seq)
            n_frame += nf
            timer_avgs.append(ta)
            timer_calls.append(tc)

        #"""
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type, step=opt.step_size, fut=opt.future_len)

        #get info
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        height = int(re.findall('imHeight=\d+', meta_info)[0].replace('imHeight=', ''))
        width = int(re.findall('imWidth=\d+', meta_info)[0].replace('imWidth=', ''))

        acc, data_dict = evaluator.eval_file(result_filename, opt, [height, width])
        accs.append(acc)
        #'''
        if opt.lstm:
            print("!! save ", os.path.join(result_root, '{}.pkl'.format(seq)))
            with open(os.path.join(result_root, '{}.pkl'.format(seq)), 'wb') as f:
                pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))
    #"""

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if opt.static_val:
        #seqs_str = '''MOT17-02-SDP
        #              MOT17-04-SDP
        #              MOT17-09-SDP  
        #              MOT20-01  
        #              MOT20-03	
        #              MOT20-05
        #              MOT20-02'''
        seqs_str = '''MOT17-04-SDP'''
        data_root = os.path.join(opt.data_dir, 'StaticMOT/images/val')

    if opt.mot17half_val:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP  
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17_half/images/val')
    
    if opt.static_train:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-09-SDP  
                      MOT20-01  
                      MOT20-03	
                      MOT20-05
                      MOT20-02'''
        data_root = os.path.join(opt.data_dir, 'StaticMOT/images/train')

    seqs = [seq.strip() for seq in seqs_str.split()]
    main(opt,
         data_root=data_root,
         seqs=seqs,
         save_images=False,
         save_videos=False)
