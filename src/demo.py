from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.sequence_loader as datasets
from track import eval_seq

logger.setLevel(logging.INFO)
import ntpath

def demo(opt):

    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)
    logger.info('Starting tracking...')
    
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    name = opt.load_model.split('/')[-2]+'_'+ntpath.basename(opt.input_video).replace('.mp4', '')
    result_filename = os.path.join(result_root, name+'_results.txt')
    frame_rate = dataloader.frame_rate
    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, name+'_frame')

    frame_id, average_time, calls = eval_seq(opt, dataloader, 'final0110_nopast', result_filename, use_cuda=opt.gpus!=[-1])
    print(frame_id, average_time, calls)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
