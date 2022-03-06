#final0110_nopast
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
import numpy as np
import torch.nn.functional as F

class BaseTrainer(object):
  def __init__(self, opt, model, lstm_model, encoder, optimizer=None, optimizer_lstm=None):
    self.lstm_model = lstm_model
    self.bbox_model = model
    self.encoder = encoder
    self.opt = opt
    self.optimizer = optimizer
    self.optimizer_lstm = optimizer_lstm #for seprate backprop
    self.loss_stats, self.bbox_loss, self.future_loss = self._get_losses(opt)

    #for warmup, only need for t-1
    self.past_info_label = None

    #for future flow
    self.warmup = True
    #self.hidden_state = torch.cuda.FloatTensor(opt.batch_size, 128, 152, 272).fill_(0)

    #add for learn uncertainty parameters
    self.optimizer.add_param_group({'params': self.bbox_loss.parameters()})
    if self.optimizer_lstm is not None:
      self.optimizer_lstm.add_param_group({'params': self.future_loss.parameters()})
    else:
      self.optimizer.add_param_group({'params': self.future_loss.parameters()})

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.lstm_model = DataParallel(self.lstm_model, device_ids=gpus,chunk_sizes=chunk_sizes).to(device)
      self.future_loss = DataParallel(self.future_loss, device_ids=gpus,chunk_sizes=chunk_sizes).to(device)
      self.bbox_model = DataParallel(self.bbox_model, device_ids=gpus,chunk_sizes=chunk_sizes).to(device)
      self.bbox_loss = DataParallel(self.bbox_loss, device_ids=gpus,chunk_sizes=chunk_sizes).to(device)
      self.encoder = DataParallel(self.encoder, device_ids=gpus,chunk_sizes=chunk_sizes).to(device)
    else:
      self.lstm_model = self.lstm_model.to(device)
      self.future_loss = self.future_loss.to(device)
      self.bbox_model = self.bbox_model.to(device)
      self.bbox_loss = self.bbox_loss.to(device)
      self.encoder = self.encoder.to(device)

    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

    if self.optimizer_lstm is not None:
      for state in self.optimizer_lstm.state.values():
        for k, v in state.items():
          if isinstance(v, torch.Tensor):
            state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    lstm_model = self.lstm_model
    bbox_model = self.bbox_model

    #freeze id features if it is over 20 epochs
    if epoch > 20:
      for name, param in bbox_model.named_parameters():
        if name in ['module.id.0.weight','module.id.0.bias','module.id.2.weight','module.id.2.bias']:
          param.requires_grad = False

    encoder = self.encoder
    opt = self.opt

    if phase == 'train':
      lstm_model.train()
      bbox_model.train()
      encoder.train()
    else:
      lstm_model.eval()
      bbox_model.eval()
      encoder.eval()
      torch.cuda.empty_cache()

    #logging
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}

    First = True #for first batch, every batch has its own random augmentation
    num_iters = 0

    #training
    for iter_id, batch in enumerate(data_loader):
      #check dataloader
      '''
      debug = False
      if debug:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for b, images in enumerate(batch['input']):
          for f, image in enumerate(images): #3, 608, 1088
            test = image.permute(1, 2, 0)
            #print(image.shape)
            plt.imshow(test)
            plt.show()
            plt.savefig('/home/u4851006/My_FairMoT_1220/demos/label'+'_b'+str(b)+'_f'+str(f)+'_i'+str(iter_id)+'.jpg')
            plt.clf()
      '''
      if First:
        First = False
        num_iters = len(data_loader)
        print('num_iters', num_iters)
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()

      data_time.update(time.time() - end)
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)

      outputs,features = bbox_model(batch['input'][:,0]) #run detection

      if torch.any(batch['flag'].data > 0): #lstm
        print(" lstm")

        #seperate backprop
        if self.optimizer_lstm is not None:
          features = features.detach()

        #run encoder

        #retain for later encoder backprop from future loss, 0112
        if self.warmup == True: #not first time in lstm
          self.hidden_state = self.hidden_state.detach() #do not backprop past graphs
        else:
          self.warmup = True
        #self.warmup = True
        #self.hidden_state = self.hidden_state.detach()
        
        state = encoder(self.hidden_state,features,torch.zeros(1))
        #state, encoded, det = encoder(self.hidden_state,features)
        self.hidden_state = state
        #run lstm
        #future_predict = lstm_model(state, encoded, p_delta_flow, self.past_info_label['ind'][:,0])
        future_predict = lstm_model(state) #p_delta_flow is t~t-1 warp, just flip it to create future estimate
        loss, loss_stats = self.future_loss(outputs, batch, future_predict=future_predict, index_list=range(opt.future_len), past_info=self.past_info_label)
        
        #past info
        self.past_info_label = batch

        if phase == 'train':
          if self.optimizer_lstm is not None: 
            self.optimizer.zero_grad()
            self.optimizer_lstm.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_lstm.step()
          else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

      else: #warmup
        print(" warmup")
        #retain for later encoder backprop from future loss, 0112
        features = features.detach()
        if self.warmup: #first
          self.hidden_state = torch.cuda.FloatTensor(opt.batch_size, 128, 152, 272).fill_(0)
          self.warmup = False
          #run encoder
          state = encoder(self.hidden_state,features,torch.ones(1))
          #p_delta_flow = None
        else:
          #run encoder
          #self.hidden_state = self.hidden_state.detach() #do not backprop past graphs
          state = encoder(self.hidden_state,features,torch.zeros(1))

        self.hidden_state = state
        #calculate loss and backprop
        loss, loss_stats = self.bbox_loss(outputs, batch, past_info=self.past_info_label)
        self.past_info_label = batch
        
        if phase == 'train':
          self.optimizer.zero_grad()
          loss.backward(retain_graph=True) #retain for later encoder backprop from future loss, 0112
          #loss.backward()
          self.optimizer.step()

      #logging
      #To Do, save future predict output
      output = outputs[-1]
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'][:,0].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].val)

        #if l == 'future_loss' or l == 'p_warp_loss':
        #Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].val)
        #else:
        #Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Time {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats, batch
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results

  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)