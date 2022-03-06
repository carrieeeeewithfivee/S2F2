from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

#from .bbox_networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .bbox_networks.my_pose_dla_dcn import get_pose_net as get_dla_dcn

from .lstm_networks.gru_decoder0110_nopast import get_gru_decoder as get_gru_decoder_0110_nopast
from .lstm_networks.gru_decoder0114_past import get_gru_decoder as get_gru_decoder_0114_past
from .lstm_networks.gru_decoder0207_ablationpast import get_gru_decoder as get_gru_decoder0207_ablationpast

from .lstm_networks.gru_encoder0110_nopast import get_gru_encoder as get_gru_encoder_0110_nopast
from .lstm_networks.gru_encoder0114_past import get_gru_encoder as get_gru_encoder_0114_past
from .lstm_networks.gru_encoder0207_ablationpast import get_gru_encoder as get_gru_encoder0207_ablationpast

_model_factory = {
  'dla': get_dla_dcn,
}

_decoder_factory = {
  'gru_decoder_0110_nopast': get_gru_decoder_0110_nopast,
  'gru_decoder0114_past': get_gru_decoder_0114_past,
  'gru_decoder0207_ablationpast': get_gru_decoder0207_ablationpast,
}

_encoder_factory = {
  'gru_encoder_0110_nopast': get_gru_encoder_0110_nopast,
  'gru_encoder0114_past': get_gru_encoder_0114_past,
  'gru_encoder0207_ablationpast': get_gru_encoder0207_ablationpast,
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def create_decoder(future_len, arch='sfh_gru'):
  get_model = _decoder_factory[arch]
  lstm_model = get_model(future_len=future_len)
  return lstm_model

def create_encoder(arch='gru_encoder'):
  get_model = _encoder_factory[arch]
  encoder = get_model()
  return encoder

def load_model(bbox_model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None, lstm_model=None, optimizer_lstm=None, encoder=None):
  
  #load pretrained weights or resume bbox model
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  
  model_state_dict = bbox_model.state_dict()
  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
        
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]

  #'hm.0.weight', 'hm.0.bias', 'hm.2.weight', 'hm.2.bias', 'wh.0.weight', 'wh.0.bias', 'wh.2.weight', 'wh.2.bias', 'id.0.weight', 'id.0.bias', 'id.2.weight', 'id.2.bias', 'reg.0.weight', 'reg.0.bias', 'reg.2.weight', 'reg.2.bias'
  #load hm, id, reg encoder parameters
  """
  encoder_state_dict = encoder.state_dict()
  e_state_dict = {}
  for k in state_dict:
    if k in encoder_state_dict:
      if state_dict[k].shape == encoder_state_dict[k].shape:
        e_state_dict[k] = state_dict[k]
      else:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}.'.format(k, state_dict[k].shape, encoder_state_dict[k].shape))
  """
  #load parameters
  bbox_model.load_state_dict(state_dict, strict=False)
  if 'lstm_model' in checkpoint and lstm_model is not None: #load decoder
    lstm_model.load_state_dict(checkpoint['lstm_model'])
  if 'encoder' in checkpoint and encoder is not None: #load encoder
    encoder.load_state_dict(checkpoint['encoder'])
  else:
    encoder.load_state_dict(e_state_dict, strict=False)


  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)

    if 'optimizer_lstm' in checkpoint and optimizer_lstm is not None:
        optimizer_lstm.load_state_dict(checkpoint['optimizer_lstm'])
    else:
      print('No optimizer parameters in checkpoint.')
  
  return bbox_model, optimizer, lstm_model, optimizer_lstm, start_epoch, encoder #if dont have will be none

def save_model(path, epoch, model, optimizer=None, lstm_model=None, optimizer_lstm=None, encoder=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict,
          'lstm_model': lstm_model.state_dict(),
          'encoder': encoder.state_dict()}
  
  if not (optimizer_lstm is None):
    data['optimizer_lstm'] = optimizer_lstm.state_dict()
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()

  torch.save(data, path)

