from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('task', default='only_flow', help='original mot, for choosing loss and trainer')
    self.parser.add_argument('--dataset', default='jde', help='jde')
    self.parser.add_argument('--exp_id', default='default')
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 
    # forecasting settings
    self.parser.add_argument('--seq_len', type=int, default=5, help='lenth of windowed sequence after warmup')
    self.parser.add_argument('--future_len', type=int, default=3, help='lenth of predicted future sequence, not including current image')
    self.parser.add_argument('--warmup_len', type=int, default=3, help='warm up not including current image') #2
    self.parser.add_argument('--step_size', type=int, default=10, help='step size') #should be in sync with fps

    self.parser.add_argument('--decoder_arch', type=str,default='sfh_gru', help='choose lstm model')
    self.parser.add_argument('--encoder_arch', type=str,default='gru_encoder', help='choose encoder model')
    self.parser.add_argument('--f_weight', nargs='+', type=float, default=[0.34,0.33,0.33], help='len = future_len, weight of f_loss')

    #self.parser.add_argument('--warp_weight', type=float, default=10, help='past warp loss weight')
    #self.parser.add_argument('--mask', action='store_true', help='mask past encoder flow')
    #self.parser.add_argument('--mask2', action='store_true', help='mask future decoder flow')

    self.parser.add_argument('--seperate_backprop', action='store_true', help='whether backprop back')
    self.parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    self.parser.add_argument('--num_epochs', type=int, default=30, help='total training epochs.')

    #demo settings
    self.parser.add_argument('--polygon', action='store_true', help='future frames show polygon or boxes')
    self.parser.add_argument('--save_flow', action='store_true', help='save flow images')
    self.parser.add_argument('--vis_limit', action='store_true', help='limit visulization images')
    #validation settings
    self.parser.add_argument('--exp_name', type=str, default='val_name',help='save to val name')
    self.parser.add_argument('--run_model', action='store_true', help='re-run model if not run before')
    self.parser.add_argument('--lstm', action='store_true', help='validate future')
    #sequence datasets for validation
    self.parser.add_argument('--static_val', default=False, help='val static_val')
    self.parser.add_argument('--static_train', default=False, help='val static_train')
    self.parser.add_argument('--mot17half_val', default=False, help='val mot17half_val')

    #Demos
    self.parser.add_argument('--input-video', type=str,
                             default='../videos/MOT16-03.mp4',
                             help='path to the input video')
    self.parser.add_argument('--output-format', type=str, default='video', help='video or text')
    self.parser.add_argument('--output-root', type=str, default='../demos', help='expected output root path')

    # system
    self.parser.add_argument('--gpus', default='0, 1',
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=8,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet

    # log
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    self.parser.add_argument('--vis_thresh', type=float, default=0.5,
                             help='visualization threshold.')
    
    # model
    self.parser.add_argument('--arch', default='dla_34', 
                             help='model architecture. Currently tested'
                                  'resdcn_34 | resdcn_50 | resfpndcn_34 |'
                                  'dla_34 | hrnet_18')
    self.parser.add_argument('--head_conv', type=int, default=-1,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '256 for resnets and 256 for dla.')
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')

    # input
    self.parser.add_argument('--input_res', type=int, default=-1, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=-1, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=-1, 
                             help='input width. -1 for default from dataset.')
    
    # train
    self.parser.add_argument('--lr', type=float, default=1e-4,
                             help='learning rate for batch size 12.')
    self.parser.add_argument('--lr_step', type=str, default='20',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--val_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    self.parser.add_argument('--trainval', action='store_true',
                             help='include validation in training and '
                                  'test on test set')

    # test
    self.parser.add_argument('--K', type=int, default=500,
                             help='max number of output objects.') 
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')
    self.parser.add_argument('--fix_res', action='store_true',
                             help='fix testing resolution or keep '
                                  'the original resolution')
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')
    # tracking
    self.parser.add_argument('--test_mot16', default=False, help='test mot16')
    self.parser.add_argument('--val_mot15', default=False, help='val mot15')
    self.parser.add_argument('--test_mot15', default=False, help='test mot15')
    self.parser.add_argument('--val_mot16', default=False, help='val mot16 or mot15')
    self.parser.add_argument('--test_mot17', default=False, help='test mot17')
    self.parser.add_argument('--val_mot17', default=False, help='val mot17')
    self.parser.add_argument('--val_mot20', default=False, help='val mot20')
    self.parser.add_argument('--test_mot20', default=False, help='test mot20')
    self.parser.add_argument('--val_hie', default=False, help='val hie')
    self.parser.add_argument('--test_hie', default=False, help='test hie')
    self.parser.add_argument('--conf_thres', type=float, default=0.4, help='confidence thresh for tracking')
    self.parser.add_argument('--det_thres', type=float, default=0.3, help='confidence thresh for detection')
    self.parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresh for nms')
    self.parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    self.parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')

    # mot
    self.parser.add_argument('--data_cfg', type=str,
                             default='../src/lib/cfg/data.json',
                             help='load data from cfg')
    self.parser.add_argument('--data_dir', type=str, default='/work/u4851006/TrackingDataset/dataset')

    # loss
    self.parser.add_argument('--mse_loss', action='store_true',
                             help='use mse loss or focal loss to train '
                                  'keypoint heatmaps.')

    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    self.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for bounding box size.')
    self.parser.add_argument('--id_loss', default='ce',
                             help='reid loss: ce | triplet')
    self.parser.add_argument('--id_weight', type=float, default=1,
                             help='loss weight for id')
    self.parser.add_argument('--reid_dim', type=int, default=128,
                             help='feature dim for reid')
    self.parser.add_argument('--ltrb', default=True,
                             help='regress left, top, right, bottom of bbox')

    self.parser.add_argument('--norm_wh', action='store_true',
                             help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
    self.parser.add_argument('--dense_wh', action='store_true',
                             help='apply weighted regression near center or '
                                  'just apply regression on center point.')
    self.parser.add_argument('--cat_spec_wh', action='store_true',
                             help='category specific bounding box size.')
    self.parser.add_argument('--not_reg_offset', action='store_true',
                             help='not regress local offset.')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 256
    opt.pad = 31
    opt.num_stacks = 1

    if opt.trainval:
      opt.val_intervals = 100000000

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')
    return opt

  def update_dataset_info_and_set_heads(self, opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.mean, opt.std = dataset.mean, dataset.std
    opt.num_classes = dataset.num_classes

    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
    
    opt.heads = {'hm': opt.num_classes,
                 'wh': 2 if not opt.ltrb else 4,
                 'id': opt.reid_dim}
    #opt.heads = {'hm': opt.num_classes, #not predict id #for bytetrack with no id
    #             'wh': 2 if not opt.ltrb else 4,}
    if opt.reg_offset:
      opt.heads.update({'reg': 2})

    opt.nID = dataset.nID
    opt.img_size = (1088, 608)
  
    print('heads', opt.heads)
    return opt

  def init(self, args=''):
    default_dataset_info = {
      'final0110_nopast': {'default_resolution': [608, 1088], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'jde', 'nID': 14455},
      'final0114_past': {'default_resolution': [608, 1088], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'jde', 'nID': 14455},
      'final0207_ablationpast': {'default_resolution': [608, 1088], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'jde', 'nID': 14455},
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    dataset = Struct(default_dataset_info[opt.task])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt
