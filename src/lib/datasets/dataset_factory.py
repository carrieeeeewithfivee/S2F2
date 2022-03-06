from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.sequence_loader import JointDataset
from .dataset.sequence_loader import RandomSequenceBatchSampler

def get_dataset(dataset, task):
  return JointDataset,RandomSequenceBatchSampler
  
