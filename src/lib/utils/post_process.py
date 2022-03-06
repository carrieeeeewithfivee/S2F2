from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

#future box process by yuwen
def bbox_post_process(_bbox, c, s, h, w, num_classes):
  pred = []
  for i in range(_bbox.shape[0]):
    _bbox[i, :, :2] = transform_preds(_bbox[i, :, 0:2], c[i], s[i], (w, h)).astype(np.float32)
    _bbox[i, :, 2:4] = transform_preds(_bbox[i, :, 2:4], c[i], s[i], (w, h)).astype(np.float32)
    pred.append(_bbox[0])
  #_bbox = _bbox[0] #no multi label
  return pred

#not implemented multiple classes
def f_ctdet_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x dim
  for i in range(dets.shape[0]):
    dets[i, :, :2] = transform_preds(dets[i, :, :2], c, s, (w, h)).astype(np.float32)
    dets[i, :, 2:] = transform_preds(dets[i, :, 2:], c, s, (w, h)).astype(np.float32)
  return dets
