import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext
from torchtext.vocab import Vectors, GloVe
import itertools, os 
import tempfile, subprocess

use_gpu = torch.cuda.is_available()

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

def moses_multi_bleu(outputs, references, lw=False):
    '''Outputs, references are lists of strings. Calculates BLEU score using https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl -- Python function from Google '''
    
    # Save outputs and references as temporary text files
    out_file = tempfile.NamedTemporaryFile()
    out_file.write('\n'.join(outputs).encode('utf-8'))
    out_file.write(b'\n')
    out_file.flush() #?
    ref_file = tempfile.NamedTemporaryFile()
    ref_file.write('\n'.join(references).encode('utf-8'))
    ref_file.write(b'\n')
    ref_file.flush() #?
    
    # Use moses multi-bleu script
    with open(out_file.name, 'r') as read_pred:
        bleu_cmd = ['scripts/multi-bleu.perl']
        bleu_cmd = bleu_cmd + ['-lc'] if lw else bleu_cmd
        bleu_cmd = bleu_cmd + [ref_file.name]
        try: 
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode('utf-8')
            #print(bleu_out)
            bleu_score = float(re.search(r'BLEU = (.+?),', bleu_out).group(1))
        except subprocess.CalledProcessError as error:
            print(error)
            raise Exception('Something wrong with bleu script')
            bleu_score = 0.0
    
    # Close temporary files
    out_file.close()
    ref_file.close()
   
    return bleu_score

