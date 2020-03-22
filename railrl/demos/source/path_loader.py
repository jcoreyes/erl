from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.torch_rl_algorithm import TorchTrainer

from railrl.misc.asset_loader import load_local_or_remote_file

import random
from railrl.torch.core import np_to_pytorch_batch
from railrl.data_management.path_builder import PathBuilder

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from railrl.core import logger

import glob

class PathLoader:
    """
    Loads demonstrations and/or off-policy data into a Trainer
    """

    def load_demos(self, ):
        pass
