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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .collections import AttrDict

__C = AttrDict()
config = __C

__C.DEBUG = False

# Training options
__C.DATASET = AttrDict()
__C.DATASET.NAME = ""
__C.DATASET.DIR  = ""

# Training options
__C.TRAIN = AttrDict()
__C.TRAIN.PARAMS_FILE = ''
__C.TRAIN.DATA_TYPE = 'train'
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.SHUFFLE = False
# scale/ar augmeantion
__C.TRAIN.CROP_SIZE = 224

# Number of iterations after which model should be tested on test/val data
__C.TRAIN.EVAL_PERIOD = 1000
__C.TRAIN.DATASET_SIZE = 234643
__C.TRAIN.DROPOUT_RATE = 0.0
__C.TRAIN.TEST_AFTER_TRAIN = False

# Train model options
__C.MODEL = AttrDict()
__C.MODEL.NUM_CLASSES = -1
__C.MODEL.MODEL_NAME = ''

# vgg
__C.MODEL.BACKBONE = ""
__C.MODEL.HEAD=""
__C.MODEL.INCHANNEL=[256, 512, 512]
__C.MODEL.SCALES=[1]
__C.MODEL.MODE=""
__C.MODEL.USE_DROPOUT=False
__C.MODEL.DROPOUT_RATE=0.5

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

# for ResNet or ResNeXt only
__C.RESNETS = AttrDict()
__C.RESNETS.NUM_GROUPS = 1
__C.RESNETS.WIDTH_PER_GROUP = 64
__C.RESNETS.STRIDE_1X1 = False
__C.RESNETS.TRANS_FUNC = 'bottleneck_transformation'


# Test
__C.TEST = AttrDict()
__C.TEST.PARAMS_FILE = ''
__C.TEST.DATA_TYPE = ''
__C.TEST.BATCH_SIZE = 64
__C.TEST.SCALE = 256
__C.TEST.CROP_SIZE = 224

# Solver
__C.SOLVER = AttrDict()
__C.SOLVER.NESTEROV = True
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
__C.SOLVER.WARMUP.WARMUP_END_ITER = 5005 * 5  # 5 epochs

# Checkpoint options
__C.CHECKPOINT = AttrDict()
__C.CHECKPOINT.CHECKPOINT_MODEL = True
__C.CHECKPOINT.RESUME = True
__C.CHECKPOINT.DIR = './out/checkpoints/'

# Checkpoint options
__C.TENSORBOARD = AttrDict()
__C.TENSORBOARD.DIR = './out/tensorboard/'

# Non-local Block
__C.NONLOCAL = AttrDict()
__C.NONLOCAL.USE_SOFTMAX = True
__C.NONLOCAL.USE_BN = True
__C.NONLOCAL.USE_AFFINE = False

__C.NONLOCAL.BN_MOMENTUM = 0.9
__C.NONLOCAL.BN_EPSILON = 1.0000001e-5
__C.NONLOCAL.BN_INIT_GAMMA = 0.0



__C.NUM_GPUS = 8

def print_cfg():
    import pprint
   

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
        if type(value) is dict:
            value = AttrDict(value)
        if key not in dict_b:
            dict_b[key]= value
            # raise KeyError('Invalid key in config file: {}'.format(key))
        if type(value) is dict:
            dict_a[key] = value 
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
        if isinstance(value, dict):
            dict_b[key] = AttrDict(value)
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