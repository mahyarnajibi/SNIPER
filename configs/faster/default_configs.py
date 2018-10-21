# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# by Mahyar Najibi
# --------------------------------------------------------------

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()
config.proposal_path = 'data/proposals'
config.MXNET_VERSION = ''

config.output_path = ''
config.symbol = ''
config.gpus = ''
config.CLASS_AGNOSTIC = True
# default training
config.default = edict()
config.default.kvstore = 'device'

# network related params
config.network = edict()
config.network.pretrained = ''
config.network.pretrained_epoch = 0
config.network.PIXEL_MEANS = np.array([0, 0, 0])
config.network.RPN_FEAT_STRIDE = 16
config.network.FIXED_PARAMS = ['gamma', 'beta']
config.network.ANCHOR_SCALES = (8, 16, 32)
config.network.ANCHOR_RATIOS = (0.5, 1, 2)
config.network.NUM_ANCHORS = len(config.network.ANCHOR_SCALES) * len(config.network.ANCHOR_RATIOS)

# dataset related params
config.dataset = edict()
config.dataset.dataset = 'PascalVOC'
config.dataset.image_set = '2007_trainval'
config.dataset.test_image_set = '2007_test'
config.dataset.root_path = './data'
config.dataset.dataset_path = './data/VOCdevkit'
config.dataset.NUM_CLASSES = 21


config.TRAIN = edict()
config.TRAIN.ONLY_PROPOSAL = False
config.TRAIN.CPP_CHIPS = False
config.TRAIN.USE_NEG_CHIPS = True
config.TRAIN.CHIPS_DB_PARTS = 20
config.TRAIN.WITH_MASK = False
config.TRAIN.lr = 0
config.TRAIN.VALID_RANGES = ((-1, 80), (32, 150), (120, -1))
config.TRAIN.SCALES = (3.0, 1.667, 512.0)

config.TRAIN.lr_step = ''
config.TRAIN.scale = 1.0
config.TRAIN.lr_factor = 0.1
config.TRAIN.warmup = False
config.TRAIN.warmup_lr = 0
config.TRAIN.warmup_step = 0
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 0.0005
config.TRAIN.fp16 = False
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 0
config.TRAIN.model_prefix = ''

config.TRAIN.ALTERNATE = edict()
config.TRAIN.ALTERNATE.RPN_BATCH_IMAGES = 0
config.TRAIN.ALTERNATE.RCNN_BATCH_IMAGES = 0
config.TRAIN.ALTERNATE.rpn1_lr = 0
config.TRAIN.ALTERNATE.rpn1_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn1_epoch = 0       # recommend 3
config.TRAIN.ALTERNATE.rfcn1_lr = 0
config.TRAIN.ALTERNATE.rfcn1_lr_step = ''   # recommend '5'
config.TRAIN.ALTERNATE.rfcn1_epoch = 0      # recommend 8
config.TRAIN.ALTERNATE.rpn2_lr = 0
config.TRAIN.ALTERNATE.rpn2_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn2_epoch = 0       # recommend 3
config.TRAIN.ALTERNATE.rfcn2_lr = 0
config.TRAIN.ALTERNATE.rfcn2_lr_step = ''   # recommend '5'
config.TRAIN.ALTERNATE.rfcn2_epoch = 0      # recommend 8
# optional
config.TRAIN.ALTERNATE.rpn3_lr = 0
config.TRAIN.ALTERNATE.rpn3_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn3_epoch = 0       # recommend 3

# whether flip image
config.TRAIN.FLIP = True
# whether shuffle image
config.TRAIN.SHUFFLE = True
# whether use OHEM
config.TRAIN.ENABLE_OHEM = False
# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 2
# e2e changes behavior of anchor loader and metric
config.TRAIN.END2END = False


# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 128
config.TRAIN.BATCH_ROIS_OHEM = 128
# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])
config.TRAIN.visualization_path = 'debug/visualization'
config.TRAIN.visualization_freq= 100
# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# used for end2end training
# RPN proposal
config.TRAIN.CXX_PROPOSAL = True
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 2000
config.TRAIN.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE
# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

config.TEST = edict()
config.TEST.NMS_SIGMA = 0.6
config.TEST.TEST_FLAG = False
# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 300
config.TEST.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000
config.TEST.PROPOSAL_MIN_SIZE = config.network.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.3

config.TEST.max_per_image = 300

# Test Model Epoch
config.TEST.test_epoch = 0


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'TRAIN':
                        if 'BBOX_WEIGHTS' in v:
                            v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
                    elif k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == 'SCALES':
                        if type(v)!=list:
                            config[k][0] = (tuple(v))
                        else:
                            config[k] = v
                    else:
                        config[k] = v
            else:
                raise ValueError("key must exist in config.py")


def update_config_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
