# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
sys.path.append('..')
from utils import DotDict, namedtuple_with_defaults, zip_namedtuple, config_as_dict

RandCropper = namedtuple_with_defaults('RandCropper',
    'min_crop_scales, max_crop_scales, \
    min_crop_aspect_ratios, max_crop_aspect_ratios, \
    min_crop_overlaps, max_crop_overlaps, \
    min_crop_sample_coverages, max_crop_sample_coverages, \
    min_crop_object_coverages, max_crop_object_coverages, \
    max_crop_trials',
    [0.0, 1.0,
    1.0, 1.0,
    0.0, 1.0,
    0.0, 1.0,
    0.0, 1.0,
    50])

RandPadder = namedtuple_with_defaults('RandPadder',
    'rand_pad_prob, max_pad_scale, fill_value',
    [0.0, 1.0, 127])

ColorJitter = namedtuple_with_defaults('ColorJitter',
    'random_hue_prob, max_random_hue, \
    random_saturation_prob, max_random_saturation, \
    random_illumination_prob, max_random_illumination, \
    random_contrast_prob, max_random_contrast',
    [0.0, 0,
    0.0, 0,
    0.0, 0,
    0.0, 0.0])


cfg = DotDict()
cfg.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# training configs
cfg.train = DotDict()
# # random cropping samplers
cfg.train.color_jitter = ColorJitter()
cfg.train.rand_crop_samplers = []
# # random padding
cfg.train.rand_pad = RandPadder(rand_pad_prob=0.0, max_pad_scale=1.0)
cfg.train.rand_crop_prob = 1.0
cfg.train.preprocess_threads = 48
cfg.train.shuffle = True
cfg.train.rand_mirror_prob = 0.5
cfg.train.crop_emit_mode = 'center'
cfg.train.num_crop_sampler = 5
cfg.train.inter_method = 10
cfg.train.random_hue_prob = 0.125
cfg.train.random_saturation_prob = 0.125
cfg.train.random_illumination_prob = 0.125
cfg.train.min_crop_scales = 0.3
cfg.train.seed = 233
cfg.train = config_as_dict(cfg.train)  # convert to normal dict


# validation
cfg.valid = DotDict()
cfg.valid.rand_crop_samplers = []
cfg.valid.rand_pad = RandPadder()
cfg.valid.color_jitter = ColorJitter()
cfg.valid.rand_mirror_prob = 0
cfg.valid.shuffle = False
cfg.valid.seed = 0
cfg.valid.preprocess_threads = 32
cfg.valid = config_as_dict(cfg.valid)  # convert to normal dict
