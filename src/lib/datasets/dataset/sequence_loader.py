import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch
import copy

from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from cython_bbox import bbox_overlaps as bbox_ious
from opts import opts
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta

from torch.utils.data.sampler import Sampler

class RandomSequenceBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, wup=3, win=5, f_len=3, step=10):
        self.nds = dataset.nds #number of datas in sequence
        self.cds = dataset.cds #start index

        self.wup = wup #warmup lenth
        self.fen = f_len #future lenth
        self.win = win #window data size
        self.batch_size = batch_size
        self.step = step
        self.first = True
        random.seed(10) #for reproduceing

        #create windowed sequence
        self.sequence = []
        for key in range(len(self.nds)):
            num = int(self.nds[key])-(self.fen)*self.step #remove future len * step
            start = int(self.cds[key])
            seq_len = (self.wup + self.win)*self.step #actual len is (self.wup + self.win)
            seq_num = ((num-self.wup*self.step)//(self.win*self.step)) #actual num * step
            seq_flag = [0]*self.wup+[1]*self.win
            for i in range(seq_num):
                for j in range(self.step):
                    rs = random.random() #for same augmentation for each image sequence
                    self.sequence.append([[f,rs,i,self.step] for f,i in zip(seq_flag,range(start+j,start+seq_len+j,self.step))])
                start = start+seq_len-self.wup*self.step
        self.lenth = (len(self.sequence)//self.batch_size)*self.batch_size*(self.wup+self.win)//self.batch_size        

    def __iter__(self):
        if not self.first:
            #create windowed sequence if not first iteration
            self.sequence = []
            for key in range(len(self.nds)):
                num = int(self.nds[key])-(self.fen)*self.step #remove future len * step
                start = int(self.cds[key])
                seq_len = (self.wup + self.win)*self.step #actual len is (self.wup + self.win)
                seq_num = ((num-self.wup*self.step)//(self.win*self.step)) #actual num * step
                seq_flag = [0]*self.wup+[1]*self.win
                for i in range(seq_num):
                    for j in range(self.step):
                        rs = random.random() #for same augmentation for each image sequence
                        self.sequence.append([[f,rs,i,self.step] for f,i in zip(seq_flag,range(start+j,start+seq_len+j,self.step))])
                    start = start+seq_len-self.wup*self.step
            self.lenth = (len(self.sequence)//self.batch_size)*self.batch_size*(self.wup+self.win)//self.batch_size
        else:
            self.first = False

        random.shuffle(self.sequence)
        seq_len = (len(self.sequence)//self.batch_size)*self.batch_size
        self.sequence = self.sequence[:seq_len] #remove excess
        seq_batch = np.array_split(self.sequence, self.batch_size) #split into chunks
        seq_batch = np.array(seq_batch)

        _, nb, sl, _ = seq_batch.shape #(4, 47, 12, 3)
        batch = []
        for i in range(nb): #number of batches
            for j in range(sl): #sequence lenth
                batch.append(seq_batch[:, i, j, :]) #(4, 3)
        assert self.lenth == len(batch)
        return iter(batch)

    def __len__(self):
        return self.lenth

class LoadImages:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files

class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = 1920, 1080
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        #cv2.imwrite('test.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image

        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(self, path, img_size=(1088, 608), augment=False, transforms=None):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_paths, label_paths, random_seed):
        labels = []
        images = []
        #random.seed(random_seed) # for using same augmentation on same data sequences
        height = self.height
        width = self.width
        for img_path, label_path in zip(img_paths, label_paths):
            random.seed(random_seed) # for using same augmentation on same data sequences
            img = cv2.imread(img_path)  # BGR
            if img is None:
                raise ValueError('File corrupt {}'.format(img_path))
            augment_hsv = True
            if self.augment and augment_hsv:
                # SV augmentation by 50%
                fraction = 0.50
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            h, w, _ = img.shape
            img, ratio, padw, padh = letterbox(img, height=height, width=width)

            #load labels
            if os.path.isfile(label_path):
                label0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
                # Normalized xywh to pixel xyxy format
                label = label0.copy()
                label[:, 2] = ratio * w * (label0[:, 2] - label0[:, 4] / 2) + padw
                label[:, 3] = ratio * h * (label0[:, 3] - label0[:, 5] / 2) + padh
                label[:, 4] = ratio * w * (label0[:, 2] + label0[:, 4] / 2) + padw
                label[:, 5] = ratio * h * (label0[:, 3] + label0[:, 5] / 2) + padh
            else:
                label = np.array([])
            
            if self.augment:
                img, label, M = random_affine(img, label, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20), mot20=False)
            
            Debug = False
            if Debug:
                self.n = self.n+1
                from colorhash import ColorHash
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                sample1 = img.copy()
                #for label in labels:
                for box in label:
                    c = ColorHash(box[1])
                    cv2.rectangle(sample1, (box[2], box[3]), (box[4], box[5]), c.rgb, 1)
                plt.imshow(sample1)
                plt.show()
                plt.savefig('/home/u4851006/My_FairMoT_1220/demos/label'+str(self.n)+'.jpg')

            if len(label) > 0:
                # convert xyxy to xywh
                label[:, 2:6] = xyxy2xywh(label[:, 2:6].copy())  # / height
                label[:, 2] /= width
                label[:, 3] /= height
                label[:, 4] /= width
                label[:, 5] /= height
            
            if self.augment:
                if random.random() > 0.5:
                    img = np.fliplr(img)
                    label[:, 2] = 1 - label[:, 2]
            
            img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB
            
            if self.transforms is not None:
                img = self.transforms(img)

            labels.append(label)
            images.append(img)

        return images, labels

    def __len__(self):
        return self.nF  # number of batches


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def random_affine(img, target=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5), mot20=False):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if target is not None:
        if len(target) > 0:
            n = target.shape[0]
            points = target[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            if mot20:
                np.clip(xy[:, 0], 0, width,  out=xy[:, 0])
                np.clip(xy[:, 2], 0, width,  out=xy[:, 2])
                np.clip(xy[:, 1], 0, height, out=xy[:, 1])
                np.clip(xy[:, 3], 0, height, out=xy[:, 3])

            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)
            target = target[i]
            target[:, 2:6] = xy[i]

        return imw, target, M
    else:
        return imw


def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [torch.from_numpy(l) for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 6)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)


class JointDataset(LoadImagesAndLabels):  # for training
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(self, opt, root, paths, img_size=(1088, 608), augment=False, transforms=None):
        self.opt = opt
        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict() #max number of bbox id's in each sequence
        self.tid_start_index = OrderedDict() #starting number of bbox id's in each sequence
        self.num_classes = 1
        #for future prediction
        self.future_len = opt.future_len
        self.step_size = opt.step_size

        self.n = 0

        for ds, path in paths.items(): #read sequences
            sequence_paths = glob.glob(osp.join(path,'*'))
            for seq in sequence_paths:
                seqname = osp.splitext(osp.basename(seq))[0]
                with open(seq, 'r') as file:
                    self.img_files[seqname] = file.readlines()
                    self.img_files[seqname] = [osp.join(root, x.strip()) for x in self.img_files[seqname]]
                    self.img_files[seqname] = list(filter(lambda x: len(x) > 0, self.img_files[seqname]))

                self.label_files[seqname] = [
                    x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                    for x in self.img_files[seqname]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max #max persons
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)

        #self.nds = [len(x)-self.future_labels for x in self.img_files.values()] #num of photos in each sequence, remove future label numbers
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))] #start index of data
        self.nF = sum(self.nds) #returned by __len__ function

        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):
        flag, random_seed, files_index, step_size = files_index
        self.step_size = int(step_size)
        #print('!!', self.step_size)
        files_index = int(files_index)
        flag = int(flag)
        for i, c in enumerate(self.cds): #if multiple folder
            if files_index >= c:
                ds = list(self.label_files.keys())[i] #sequence name
                start_index = c

        img_paths = [self.img_files[ds][files_index - start_index + i] for i in range(0,self.future_len*self.step_size+1,self.step_size)] #multiple labels
        label_paths = [self.label_files[ds][files_index - start_index + i] for i in range(0,self.future_len*self.step_size+1,self.step_size)] #multiple labels
        imgs, labels = self.get_data(img_paths, label_paths, random_seed)
        for label in labels:
            for i, _ in enumerate(label):
                if label[i, 1] > -1:
                    label[i, 1] += self.tid_start_index[ds]

        output_h = imgs[0].shape[1] // self.opt.down_ratio #152
        output_w = imgs[0].shape[2] // self.opt.down_ratio

        #stack images
        imgs = np.stack(imgs, axis=0)
        num_classes = self.num_classes

        #for future frames, remove objects not seen in first frame
        items = [box[1] for box in labels[0]]
        assert len(labels) == self.future_len+1
        fl = self.future_len+1
        for i in range(1, fl):
            to_remove = []
            for j, box in enumerate(labels[i]):
                if box[1] not in items:
                    to_remove.append(j)
            labels[i] = np.delete(labels[i], to_remove, axis=0)

        #create input data for center net, create for each label (add to dimention 0)
        hm = np.zeros((fl, num_classes, output_h, output_w), dtype=np.float32) #heatmap
        if self.opt.ltrb: #regress left top, right bottom of bbox
            wh = np.zeros((fl, self.max_objs, 4), dtype=np.float32) #self.max_objs : testing max number of objects
        else: #regress center of bbox
            wh = np.zeros((fl, self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((fl, self.max_objs, 2), dtype=np.float32) #w,h
        ind = np.zeros((fl ,self.max_objs, ), dtype=np.int64) #index
        reg_mask = np.zeros((fl, self.max_objs, ), dtype=np.uint8)
        ids = np.zeros((fl, self.max_objs, ), dtype=np.int64)
        bbox_xys = np.zeros((fl, self.max_objs, 4), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        for i, _label in enumerate(labels):
            for k in range(len(_label)):
                label = _label[k]
                bbox = label[2:]
                cls_id = int(label[0])
                bbox[[0, 2]] = bbox[[0, 2]] * output_w
                bbox[[1, 3]] = bbox[[1, 3]] * output_h
                bbox_amodal = copy.deepcopy(bbox)
                bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
                bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
                bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
                bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
                bbox[0] = np.clip(bbox[0], 0, output_w - 1)
                bbox[1] = np.clip(bbox[1], 0, output_h - 1)
                h = bbox[3]
                w = bbox[2]

                bbox_xy = copy.deepcopy(bbox)
                bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
                bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
                bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
                bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    radius = 6 if self.opt.mse_loss else radius
                    #radius = max(1, int(radius)) if self.opt.mse_loss else radius
                    ct = np.array(
                        [bbox[0], bbox[1]], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_gaussian(hm[i, cls_id], ct_int, radius)
                    if self.opt.ltrb:
                        wh[i,k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                                bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                    else:
                        wh[i,k] = 1. * w, 1. * h
                    ind[i,k] = ct_int[1] * output_w + ct_int[0]
                    reg[i,k] = ct - ct_int
                    reg_mask[i,k] = 1
                    ids[i,k] = label[1]
                    bbox_xys[i,k] = bbox_xy

        ret = {'flag': flag, 'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys}
        return ret