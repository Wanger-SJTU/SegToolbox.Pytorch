
import argparse
import os
import time

class Options():
  def __init__(self):
    self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    self.initialized = False

  def initialize(self):
    # self.parser.add_argument('--data', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    self.parser.add_argument('--pretrain', type=str, default='', help='')
    self.parser.add_argument('--config', type=str, default='./configs/voc_vgg_fcn.yaml')
    self.parser.add_argument('--ratio', type=float, default=1)
    self.initialized = True
  
  def parse(self):
    if not self.initialized:
      self.initialize()
    opt = self.parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    # assert 0 <= opt.ratio <= 1
    args = vars(opt)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
      print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    self.opt = opt 
    return self.opt

if __name__ == '__main__':
  a = Options()
  a.parse()