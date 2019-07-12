# --------------------------------------------------------
# Configurations for domain adaptation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# Adapted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py
# --------------------------------------------------------

import os.path as osp

import numpy as np
from easydict import EasyDict

from advent.utils import project_root
from advent.utils.serialization import yaml_load


cfg = EasyDict()

# COMMON CONFIGS
# source domain
cfg.SOURCE = 'GTA'
# target domain
cfg.TARGET = 'Cityscapes'
# Number of workers for dataloading
cfg.NUM_WORKERS = 4
# List of training images
cfg.DATA_LIST_SOURCE = str(project_root / 'advent/dataset/gta5_list/{}.txt')
cfg.DATA_LIST_TARGET = str(project_root / 'advent/dataset/cityscapes_list/{}.txt')
# Directories
cfg.DATA_DIRECTORY_SOURCE = str(project_root / 'data/GTA5')
cfg.DATA_DIRECTORY_TARGET = str(project_root / 'data/Cityscapes')
# Number of object classes
cfg.NUM_CLASSES = 19
# Exp dirs
cfg.EXP_NAME = ''
cfg.EXP_ROOT = project_root / 'experiments'
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
# CUDA
cfg.GPU_ID = 0

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.SET_SOURCE = 'all'
cfg.TRAIN.SET_TARGET = 'train'
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 720)
cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 512)
# Class info
cfg.TRAIN.INFO_SOURCE = ''
cfg.TRAIN.INFO_TARGET = str(project_root / 'advent/dataset/cityscapes_list/info.json')
# Segmentation network params
cfg.TRAIN.MODEL = 'DeepLabv2'
cfg.TRAIN.MULTI_LEVEL = True
cfg.TRAIN.RESTORE_FROM = ''
cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1  # weight of conv4 prediction. Used in multi-level setting.
# Domain adaptation
cfg.TRAIN.DA_METHOD = 'AdvEnt'
# Adversarial training params
cfg.TRAIN.LEARNING_RATE_D = 1e-4
cfg.TRAIN.LAMBDA_ADV_MAIN = 0.001
cfg.TRAIN.LAMBDA_ADV_AUX = 0.0002
# MinEnt params
cfg.TRAIN.LAMBDA_ENT_MAIN = 0.001
cfg.TRAIN.LAMBDA_ENT_AUX = 0.0002
# Other params
cfg.TRAIN.MAX_ITERS = 250000
cfg.TRAIN.EARLY_STOP = 120000
cfg.TRAIN.SAVE_PRED_EVERY = 2000
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.RANDOM_SEED = 1234
cfg.TRAIN.TENSORBOARD_LOGDIR = ''
cfg.TRAIN.TENSORBOARD_VIZRATE = 100

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = 'best'  # {'single', 'best'}
# model
cfg.TEST.MODEL = ('DeepLabv2',)
cfg.TEST.MODEL_WEIGHT = (1.0,)
cfg.TEST.MULTI_LEVEL = (True,)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.RESTORE_FROM = ('',)
cfg.TEST.SNAPSHOT_DIR = ('',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_STEP = 2000  # used in 'best' mode
cfg.TEST.SNAPSHOT_MAXITER = 120000  # used in 'best' mode
# Test sets
cfg.TEST.SET_TARGET = 'val'
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_TARGET = (1024, 512)
cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
cfg.TEST.INFO_TARGET = str(project_root / 'advent/dataset/cityscapes_list/info.json')
cfg.TEST.WAIT_MODEL = True


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
