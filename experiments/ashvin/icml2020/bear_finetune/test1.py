import copy

import gym
import numpy as np
import torch.nn as nn

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.launchers.launcher_util import run_experiment
from rlkit.samplers.data_collector import MdpPathCollector # , CustomMdpPathCollector
from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.torch.networks import MlpQf, TanhMlpPolicy
from rlkit.torch.sac.policies import (
    TanhGaussianPolicy
)
from rlkit.torch.sac.bear import BEARTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rlkit.launchers.arglauncher import run_variants

import gym

from rlkit.launchers.experiments.ashvin.bear_rl import experiment, process_args

if __name__ == "__main__":
    variant = dict(
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            target_update_method='default',

            # use_automatic_entropy_tuning=True,
            # BEAR specific params
            mode='auto',
            kernel_choice='laplacian',
            policy_update_style='0',  # 0 is min, 1 is average (for double Qs)
            mmd_sigma=20.0,  # 5, 10, 40, 50

            target_mmd_thresh=0.05,  # .1, .07, 0.01, 0.02

            # gradient penalty hparams
            with_grad_penalty_v1=False,
            with_grad_penalty_v2=False,
            grad_coefficient_policy=0.001,
            grad_coefficient_q=1E-4,
            start_epoch_grad_penalty=24000,

            # Target nets/ policy vs Q-function update
            use_target_nets=True,
            policy_update_delay=1,
            num_steps_policy_update_only=1,

            ## advantage weighting
            use_adv_weighting=False,
            pretraining_env_logging_period=10000,
            pretraining_logging_period=1000,
            do_pretrain_rollouts=False,
            num_pretrain_steps=25000,
        ),
        algo_kwargs=dict(
            batch_size=256,
            max_path_length=1000,
            num_epochs=1001,
            num_eval_steps_per_epoch=3000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            # min_num_steps_before_training=10*1000,
            # max_path_length=10,
            # num_epochs=100,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
        ),
        # data_path='demos/final_buffers_buffer_medium_cheetah.npz',
        # data_path=[
        #     'demos/ant_action_noise_15.npy',
        #     'demos/ant_off_policy_15_demos_100.npy',
        # ],
        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            obs_key="state_observation",
            demo_paths=[
                # dict(
                #     path="demos/icml2020/hand/pen2_sparse.npy",
                #     obs_dict=True,
                #     is_demo=True,
                # ),
                # dict(
                #     path="demos/icml2020/hand/pen_bc_sparse4.npy",
                #     obs_dict=False,
                #     is_demo=False,
                #     train_split=0.9,
                # ),
            ],
        ),
        replay_buffer_size=int(1E6),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        expl_path_collector_kwargs=dict(render=False),
        eval_path_collector_kwargs=dict(render=False),
        shared_qf_conv=False,
        use_robot_state=True,
        batch_rl=False,

        add_env_demos=True,
        add_env_offpolicy_data=True,
    )

    search_space = {
        'env': [
            'half-cheetah',
            'ant',
            'walker',
        ],
        'shared_qf_conv': [
            True,
            # False,
        ],
        'trainer_kwargs.kernel_choice':['laplacian'],
        # 'trainer_kwargs.target_mmd_thresh':[.15, .1, .07, .05, 0.01, 0.02],
        'trainer_kwargs.target_mmd_thresh':[.05, 0.1, 0.15],
        'trainer_kwargs.num_samples_mmd_match':[10, ],
        'trainer_kwargs.mmd_sigma':[50, ],
        'seedid': range(3),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args)
