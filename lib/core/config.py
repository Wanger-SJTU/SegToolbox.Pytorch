# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
##############################################################################
#
# Based on:
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# code borrowed from 
# https://github.com/facebookresearch/video-nonlocal-net
# 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch.nn as nn
from .collections import AttrDict

__all__ = ['config', 'cfg_from_file', 'cfg_from_list']

__C = AttrDict()
config = __C

__C.DEBUG = False
__C.EXPERIMENT_NAME = ""

# Training options
__C.DATASET = AttrDict()
__C.DATASET.NAME = ""
__C.DATASET.DIR  = ""
__C.DATASET.IGNOREIDX=-1


# Training options
__C.TRAIN = AttrDict()
__C.TRAIN.PARAMS_FILE = ''
__C.TRAIN.DATA_TYPE = 'train'
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.SHUFFLE = False
# scale/ar augmeantion
__C.TRAIN.CROP_SIZE = 224
__C.TRAIN.LODEMEMORY = False

# Number of iterations after which model should be tested on test/val data
__C.TRAIN.EVAL_PERIOD = 1000
__C.TRAIN.DATASET_SIZE = 234643
__C.TRAIN.DROPOUT_RATE = 0.0
__C.TRAIN.TEST_AFTER_TRAIN = False

__C.TRAIN.LOSS = AttrDict()
__C.TRAIN.LOSS.se_loss= False
__C.TRAIN.LOSS.se_weight= 0.2
__C.TRAIN.LOSS.aux= False
__C.TRAIN.LOSS.aux_weight= 0.4
__C.TRAIN.LOSS.weight= None

# Train model options
__C.MODEL = AttrDict()
__C.MODEL.NUM_CLASSES = -1
__C.MODEL.MODEL_NAME = ''

# model
__C.MODEL.BACKBONE = ""
__C.MODEL.HEAD=""
__C.MODEL.MID=""
__C.MODEL.INCHANNEL=[]
__C.MODEL.SCALES=[]
__C.MODEL.FACTOR=1
__C.MODEL.MODE=""
__C.MODEL.USE_DROPOUT=False
__C.MODEL.DROPOUT_RATE=0.5

__C.MODEL.USE_BN=True
__C.MODEL.PRETRAIN=True
# bn
__C.MODEL.BN_MOMENTUM = 0.9
__C.MODEL.BN_EPSILON = 1.0000001e-5

# Kaiming:
# We may use 0 to initialize the residual branch of a residual block,
# so the inital state of the block is exactly identiy. 
# This helps optimizaiton.
__C.MODEL.BN_INIT_GAMMA = 1.0


# when fine-tuning with bn frozen, we turn a bn layer into affine
__C.MODEL.USE_AFFINE = False

# Non-local Block
__C.MODEL.NONLOCAL = False
__C.MODEL.JPU = False


__C.PSP = AttrDict()
__C.PSP.sizes = [1,2,3,6]
__C.PSP.deep_features_size = 256 #default
__C.PSP.drop_1 = 0.3
__C.PSP.drop_2 = 0.15

__C.NONLOCAL = AttrDict()
__C.NONLOCAL.DIM =  2 # 1D,2D,3D version

__C.NONLOCAL.TYPE= 'DotProduct' # DotProduct,LocalGaussian,LocalConcatenation,EmbeddedGaussian

__C.NONLOCAL.USE_SOFTMAX = True
__C.NONLOCAL.USE_AFFINE = False

__C.NONLOCAL.USE_BN = True
__C.NONLOCAL.BN_PARA = AttrDict()
__C.NONLOCAL.BN_PARA.momentum = 0.9
__C.NONLOCAL.BN_PARA.epsilon = 1.0000001e-5
__C.NONLOCAL.BN_PARA.init_gamma = 0.0
__C.NONLOCAL.PARA = AttrDict()
__C.NONLOCAL.PARA.index =  4 # channel for nonlocal
__C.NONLOCAL.PARA.in_channels = 1
__C.NONLOCAL.PARA.inter_channels = 512 
__C.NONLOCAL.PARA.sub_sample=True
__C.NONLOCAL.PARA.use_bn=True

__C.JPU = AttrDict()
__C.JPU.BN_PARA = AttrDict()
__C.JPU.BN_PARA.momentum = 0.9
__C.JPU.BN_PARA.epsilon = 1.0000001e-5
__C.JPU.BN_PARA.init_gamma = 0.0
__C.JPU.PARA = AttrDict()
__C.JPU.PARA.in_channels = []
__C.JPU.PARA.width=512 
__C.JPU.PARA.norm_layer = nn.BatchNorm2d
__C.JPU.PARA.up_kwargs  = AttrDict()
__C.JPU.PARA.up_kwargs.mode= 'bilinear'
__C.JPU.PARA.up_kwargs.align_corners=True

# for ResNet or ResNeXt only
__C.RESNETS = AttrDict()
__C.RESNETS.num_groups = 1
__C.RESNETS.width_per_group = 64
__C.RESNETS.stride_1x1 = False
__C.RESNETS.trans_func = 'bottleneck_transformation'

__C.RESNETS.zero_init_residual=False
__C.RESNETS.deep_base=False # trick in PSPnet
__C.RESNETS.groups=1
__C.RESNETS.output_size=8
__C.RESNETS.replace_stride_with_dilation=None
__C.RESNETS.norm_layer = nn.BatchNorm2d

# Test
__C.TEST = AttrDict()
__C.TEST.PARAMS_FILE = ''
__C.TEST.DATA_TYPE = ''
__C.TEST.BATCH_SIZE = 64
__C.TEST.SCALE = 256
__C.TEST.CROP_SIZE = 224
__C.TEST.SHUFFLE = False
__C.TEST.LODEMEMORY = False
# Solver
__C.SOLVER = AttrDict()
__C.SOLVER.OPTIM = 'adam'
__C.SOLVER.NESTEROV = True
__C.SOLVER.MAX_ITER = 1e4
__C.SOLVER.WEIGHT_DECAY = 0.0001
__C.SOLVER.WEIGHT_DECAY_BN = 0.0001
__C.SOLVER.MOMENTUM = 0.9

# Learning rates
__C.SOLVER.LR_POLICY = 'steps_with_relative_lrs'
__C.SOLVER.BASE_LR = 0.1
# For imagenet1k, 150150 = 30 epochs for batch size of 256 images over 8 gpus
__C.SOLVER.STEP_SIZES = [150150, 150150, 150150]
__C.SOLVER.LRS = [1, 0.1, 0.01]
# For above batch size, running 100 epochs = 500500 iterations
__C.SOLVER.MAX_ITER = 500500
# to be consistent with detection code, we will turn STEP_SIZES into STEPS
# example: STEP_SIZES [30, 30, 20] => STEPS [0, 30, 60, 80]
__C.SOLVER.STEPS = None
__C.SOLVER.GAMMA = 0.1  # for cfg.SOLVER.LR_POLICY = 'steps_with_decay'

__C.SOLVER.SCALE_MOMENTUM = False

# warmup hack
__C.SOLVER.WARMUP = AttrDict()
__C.SOLVER.WARMUP.WARMUP_ON = False
__C.SOLVER.WARMUP.WARMUP_START_LR = 0.1
__C.SOLVER.WARMUP.WARMUP_EPOCH = 5  # 5 epochs

# Checkpoint options
__C.CHECKPOINT = AttrDict()
__C.CHECKPOINT.CHECKPOINT_MODEL = True
__C.CHECKPOINT.RESUME = True
__C.CHECKPOINT.DIR = './out/checkpoints/'

# Checkpoint options
__C.TENSORBOARD = AttrDict()
__C.TENSORBOARD.DIR = './out/tensorboard/'
__C.TENSORBOARD.HIST = True

__C.DATALOADER = AttrDict()
__C.DATALOADER.num_workers = 4
__C.DATALOADER.pin_memory = True


BN = {
    'none'  :None,
    'bn'    :nn.BatchNorm2d,
    'syncbn':nn.SyncBatchNorm
}


def log_cfg():
    # import logging
    import time
    import os
    import yaml
    file_name = os.path.join(__C.CHECKPOINT.DIR,
                             __C.EXPERIMENT_NAME,
                             "ratio_"+str(__C.TRAIN.DROPOUT_RATE),
        time.strftime(r"%Y_%m_%d", time.localtime())+'.yaml')

    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name, 'w' ,encoding='utf8') as f:
        # f.write('------------ Configs -------------\n')
        yaml.dump(__C, f)
        # for k, v in sorted(__C.items()):
        #     f.write('%s: %s \n' % (str(k), str(v)))
        # f.write('------------ End -------------\n')

def assert_and_infer_cfg():

    # lr schedule
    if __C.SOLVER.STEPS is None:
        # example input: [150150, 150150, 150150]
        __C.SOLVER.STEPS = []
        __C.SOLVER.STEPS.append(0)
        for idx in range(len(__C.SOLVER.STEP_SIZES)):
            __C.SOLVER.STEPS.append(
                __C.SOLVER.STEP_SIZES[idx] + __C.SOLVER.STEPS[idx])
        # now we have [0, 150150, 300300, 450450]

    # we don't want to do 10-crop
    if __C.TEST.TEN_CROP:
        raise Exception('TEST.TEN_CROP is deprecated.')

    assert __C.TRAIN.BATCH_SIZE % __C.NUM_GPUS == 0, \
        "Train batch size should be multiple of num_gpus."

    assert __C.TEST.BATCH_SIZE % __C.NUM_GPUS == 0, \
        "Test batch size should be multiple of num_gpus."

def merge_dicts(dict_a, dict_b):
    from ast import literal_eval
    for key, value in dict_a.items():
        if key == 'norm_layer':
            value = BN[value]
            
        if key not in dict_b:
            raise KeyError('Invalid key in config file: {}'.format(key))
        if isinstance(value, dict):
            value = AttrDict(value) 
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        # the types must match, too
        old_type = type(dict_b[key])
        if old_type is not type(value) and value is not None:
                raise ValueError(
                    'Type mismatch ({} vs. {}) for config key: {}'.format(
                        type(dict_b[key]), type(value), key)
                )
        # recursively merge dicts
        if isinstance(value, AttrDict):
            try:
                merge_dicts(dict_a[key], dict_b[key])
            except BaseException:
                raise Exception('Error under config key: {}'.format(key))
        else:
            dict_b[key] = value

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.FullLoader))
    merge_dicts(yaml_config, __C)
 
def cfg_from_list(args_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(args_list) % 2 == 0, 'Specify values or keys for args'
    for key, value in zip(args_list[0::2], args_list[1::2]):
        key_list = key.split('.')
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, 'Config key {} not found'.format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        assert subkey in cfg, 'Config key {} not found'.format(subkey)
        try:
            # handle the case when v is a string literal
            val = literal_eval(value)
        except BaseException:
            val = value
        assert isinstance(val, type(cfg[subkey])) or cfg[subkey] is None, \
            'type {} does not match original type {}'.format(
                type(val), type(cfg[subkey]))
        cfg[subkey] = val