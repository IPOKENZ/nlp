import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext
from torchtext.vocab import Vectors, GloVe
import itertools, os 

class Logger():  
  def __init__(self):
    '''Create new log file'''
    j = 0
    while os.path.exists('saves/log-{}.log'.format(j)):
      j += 1
    self.fname = 'saves/log-{}.log'.format(j)
    
  def log(self, info, stdout=True):
    '''Print to log file and standard output'''
    with open(self.fname, 'a') as f:
      print(info, file=f)
    if stdout:
      print(info)
