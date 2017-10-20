"""
Fine tune a trained policy/qf
"""
import argparse
import random

import joblib
from pathlib import Path

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    HorizonFedStateDistanceQLearning)
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import StatConditionalSchedule
from railrl.torch.modules import HuberLoss
from railrl.torch.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
                        help='Path to snapshot file to fine tune.')
    args = parser.parse_args()

    data = joblib.load(args.path)
    algo = data['algorithm']
    algo.train()
