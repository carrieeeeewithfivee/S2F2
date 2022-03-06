import os
import numpy as np
import copy
import motmetrics as mm
mm.lap.default_solver = 'lap'
from tracking_utils.io import read_results, unzip_objs, read_f_results
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 :
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 :
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """
    x1 = 1
    y1 = 0
    x2 = 3
    y2 = 2
    assert bb1[x1] < bb1[x2]
    assert bb1[y1] < bb1[y2]
    assert bb2[x1] < bb2[x2]
    assert bb2[y1] < bb2[y2]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[x1], bb2[x1])
    y_top = max(bb1[y1], bb2[y1])
    x_right = min(bb1[x2], bb2[x2])
    y_bottom = min(bb1[y2], bb2[y2])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[x2] - bb1[x1]) * (bb1[y2] - bb1[y1])
    bb2_area = (bb2[x2] - bb2[x1]) * (bb2[y2] - bb2[y1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

class Evaluator(object):

    def __init__(self, data_root, seq_name, data_type, step=1, fut=1):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type
        self.step = step
        self.fut = fut

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        #assert self.data_type == 'mot'
        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)

    def reset_accumulator(self):
        acc = mm.MOTAccumulator(auto_id=True)
        return acc

    def eval_frame(self, acc, frame_id, trk_tlwhs, trk_ids, rtn_events=False, gt_remove=None):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        if gt_remove is not None: #remove frames not in first frame for future
            gt_start = self.gt_frame_dict.get(frame_id-self.step*gt_remove, [])
            _, items = unzip_objs(gt_start)[:2]
            to_remove = []
            for j, box in enumerate(gt_objs):
                if box[1] not in items:
                    to_remove.append(j)
            gt_objs = np.delete(gt_objs, to_remove, axis=0)
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(acc, 'last_mot_events'):
            events = acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events, acc

    def eval_future_frame(self, id_dict, frame_id, trk_tlwhs, trk_ids):
        # results
        trk_tlbrs = np.copy(trk_tlwhs)
        trk_tlbrs[:,2:] += trk_tlbrs[:,:2] #tlbr
        trk_ids = np.copy(trk_ids)
        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]
        gt_tlbrs = np.copy(gt_tlwhs)
        gt_tlbrs[:,2:] += gt_tlbrs[:,:2] #tlbr
        ious = []
        for idx, box in zip(trk_ids, trk_tlbrs):
            #use id_dict to change id
            if idx in id_dict:
                if id_dict[idx] in gt_ids:
                    gt_index = gt_ids.index(id_dict[idx])
                    #calculate iou
                    iou = get_iou(gt_tlbrs[gt_index], box)
                    ious.append(iou)
        return sum(ious)/len(ious) #return miou
    #'''
    def eval_file(self, filename, opt, info):
        acc = self.reset_accumulator()
        result_frame_dict = read_f_results(filename, self.data_type)
        frames = sorted(list(set(result_frame_dict.keys())))
        #f_accs = []
        f_mious = []
        data_dic = {} #{gtid: {frameid: [predicted box, gt box, [f_predicted boxes], [f_gt boxes]]}}
        if 'MOT20' in filename:
            mot20 = True
            h, w = info
            print(h, w)
        else:
            mot20 = False

        for frame_id in frames:
            frame_boxes = result_frame_dict.get(frame_id, dict())
            trk_objs = frame_boxes.get(0, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]

            #remove out of bound for mot20
            if mot20:
                trk_tlwhs[:,2:] += trk_tlwhs[:,:2] #tlbr
                np.clip(trk_tlwhs[:, 0], 0, w,  out=trk_tlwhs[:, 0])
                np.clip(trk_tlwhs[:, 2], 0, w,  out=trk_tlwhs[:, 2])
                np.clip(trk_tlwhs[:, 1], 0, h, out=trk_tlwhs[:, 1])
                np.clip(trk_tlwhs[:, 3], 0, h, out=trk_tlwhs[:, 3])
                #tlwh
                trk_tlwhs[:,2:] -= trk_tlwhs[:,:2] #tlbr


            events, acc = self.eval_frame(acc, frame_id, trk_tlwhs, trk_ids, rtn_events=False)
            
            if opt.lstm:
                #get tracked information, create dic to change ids
                gt_ids = acc.mot_events.loc[lambda df: df['Type'] == 'MATCH', lambda df: ['OId', 'HId']]['OId'].tolist()
                predicted_ids = acc.mot_events.loc[lambda df: df['Type'] == 'MATCH', lambda df: ['OId', 'HId']]['HId'].tolist()
                id_dict = {}
                for gt, pred in zip(gt_ids, predicted_ids):
                    id_dict[pred] = gt

                #change the ids to gt, and save to file (for running other peoples models) ToDo
                trk_tlbrs = np.copy(trk_tlwhs)
                trk_tlbrs[:,2:] += trk_tlbrs[:,:2] #tlbr
                #gts
                gt_objs = self.gt_frame_dict.get(frame_id, [])
                gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]
                gt_tlbrs = np.copy(gt_tlwhs)
                gt_tlbrs[:,2:] += gt_tlbrs[:,:2] #tlbr
                #futures
                if frame_id>30:
                    #i = self.fut
                    f_trk_tlbrs = []
                    for i in range(1,self.fut+1):
                        f_trk_obj = frame_boxes.get(i, []) #get last frame
                        f_trk_tlbr, f_trk_id = unzip_objs(f_trk_obj)[:2]
                        f_trk_tlbr[:,2:] += f_trk_tlbr[:,:2] #tlbr

                        if mot20:
                            np.clip(f_trk_tlbr[:, 0], 0, w, out=f_trk_tlbr[:, 0])
                            np.clip(f_trk_tlbr[:, 2], 0, w, out=f_trk_tlbr[:, 2])
                            np.clip(f_trk_tlbr[:, 1], 0, h, out=f_trk_tlbr[:, 1])
                            np.clip(f_trk_tlbr[:, 3], 0, h, out=f_trk_tlbr[:, 3])

                        f_trk_tlbrs.append(f_trk_tlbr)

                    #f_gt
                    f_gt_ids = []
                    f_gt_tlbrs = []
                    for i in range(1, self.fut+1):
                        #i = self.fut #get last frame
                        f_gt_obj = self.gt_frame_dict.get(frame_id+self.step*i, [])
                        f_gt_tlwh, f_gt_id = unzip_objs(f_gt_obj)[:2]
                        f_gt_tlbr = np.copy(f_gt_tlwh)
                        f_gt_tlbr[:,2:] += f_gt_tlbr[:,:2] #tlbr
                        f_gt_ids.append(f_gt_id)
                        f_gt_tlbrs.append(f_gt_tlbr)
                    
                    for f_trk_tlbr, f_gt_id, f_gt_tlbr in zip(f_trk_tlbrs, f_gt_ids, f_gt_tlbrs):
                        for idx, box, f_box in zip(trk_ids, trk_tlbrs, f_trk_tlbr):
                            #use id_dict to change id
                            if idx in id_dict:
                                if id_dict[idx] in gt_ids and id_dict[idx] in f_gt_id:
                                    f_gt_index = f_gt_id.index(id_dict[idx])
                                    f_gt_box = f_gt_tlbr[f_gt_index]
                                    if int(id_dict[idx]) in data_dic:
                                        if frame_id in data_dic[int(id_dict[idx])]:
                                            data_dic[int(id_dict[idx])][frame_id][2].append(f_box)
                                            data_dic[int(id_dict[idx])][frame_id][3].append(f_gt_box)
                                        else:
                                            gt_index = gt_ids.index(id_dict[idx])
                                            gt_box = gt_tlbrs[gt_index]
                                            data_dic[int(id_dict[idx])][frame_id] = [box, gt_box, [f_box], [f_gt_box]]
                                    else:
                                        gt_index = gt_ids.index(id_dict[idx])
                                        gt_box = gt_tlbrs[gt_index]
                                        data_dic[int(id_dict[idx])] = {frame_id: [box, gt_box, [f_box], [f_gt_box]]}

                                elif id_dict[idx] in gt_ids: #no future label
                                    gt_index = gt_ids.index(id_dict[idx])
                                    gt_box = gt_tlbrs[gt_index]
                                    if int(id_dict[idx]) in data_dic:
                                        if frame_id in data_dic[int(id_dict[idx])]:
                                            data_dic[int(id_dict[idx])][frame_id][2].append(f_box)
                                        else:
                                            data_dic[int(id_dict[idx])][frame_id] = [box, gt_box, [f_box], []]
                                    else:
                                        data_dic[int(id_dict[idx])] = {frame_id: [box, gt_box, [f_box], []]}
                else: #no future 
                    for idx, box in zip(trk_ids, trk_tlbrs):
                        #use id_dict to change id
                        if idx in id_dict:
                            if id_dict[idx] in gt_ids:
                                gt_index = gt_ids.index(id_dict[idx])
                                gt_box = gt_tlbrs[gt_index]
                                if int(id_dict[idx]) in data_dic:
                                    data_dic[int(id_dict[idx])][frame_id] = [box, gt_box]
                                else:
                                    data_dic[int(id_dict[idx])] = {frame_id: [box, gt_box]}
        return acc, data_dic
    #'''

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )
        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename, engine='openpyxl')
        summary.to_excel(writer)
        writer.save()