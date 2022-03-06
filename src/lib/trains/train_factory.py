from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .final0110_nopast.calculate_loss import MyTrainer as final0110_nopast

train_factory = {
  'final0110_nopast':final0110_nopast,
}