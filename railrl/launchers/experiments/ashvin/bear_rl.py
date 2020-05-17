import gym
from railrl.data_management.awr_env_replay_buffer import AWREnvReplayBuffer
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.wrappers import NormalizedBoxEnv, StackObservationEnv, RewardWrapperEnv
import railrl.torch.pytorch_util as ptu
from railrl.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from railrl.samplers.data_collector.step_collector import MdpStepCollector
from railrl.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from railrl.torch.sac.awr_sac import AWRSACTrainer
from railrl.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

from railrl.demos.source.hdf5_path_loader import HDF5PathLoader
from railrl.demos.source.mdp_path_loader import MDPPathLoader
from railrl.visualization.video import save_paths, VideoSaveFunction

from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.core.image_env import ImageEnv
from multiworld.core.gym_to_multi_env import GymToMultiEnv

from railrl.launchers.experiments.ashvin.rfeatures.encoder_wrapped_env import EncoderWrappedEnv
from railrl.launchers.experiments.ashvin.rfeatures.rfeatures_model import TimestepPredictionModel

import torch
import numpy as np
from torchvision.utils import save_image

from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_and_epislon import GaussianAndEpislonStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from railrl.core import logger
from railrl.misc.asset_loader import load_local_or_remote_file
import pickle

import copy
import torch.nn as nn
from railrl.samplers.data_collector import MdpPathCollector # , CustomMdpPathCollector
from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from railrl.torch.networks import MlpQf, TanhMlpPolicy
from railrl.torch.sac.policies import (
    TanhGaussianPolicy, VAEPolicy
)
from railrl.torch.sac.bear import BEARTrainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


ENV_PARAMS = {
    'half-cheetah': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'HalfCheetah-v2'
    },
    'hopper': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Hopper-v2'
    },
    'humanoid': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Humanoid-v2'
    },
    'inv-double-pendulum': {  # 2 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'InvertedDoublePendulum-v2'
    },
    'pendulum': {  # 2 DoF
        'num_expl_steps_per_train_loop': 200,
        'max_path_length': 200,
        'min_num_steps_before_training': 2000,
        'target_update_period': 200,
        'env_id':'Pendulum-v2'
    },
    'ant': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Ant-v2'
    },
    'walker': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Walker2d-v2'
    },
    'swimmer': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Swimmer-v2'
    },

    'pen-v0': {
        'env_id': 'pen-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },
    'door-v0': {
        'env_id': 'door-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },
    'relocate-v0': {
        'env_id': 'relocate-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },
    'hammer-v0': {
        'env_id': 'hammer-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },

    'pen-sparse-v0': {
        'env_id': 'pen-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/pen2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/pen_bc_sparse1.npy",
            # path="demos/icml2020/hand/pen_bc_sparse2.npy",
            # path="demos/icml2020/hand/pen_bc_sparse3.npy",
            path="demos/icml2020/hand/pen_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-sparse-v0': {
        'env_id': 'door-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/door2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/door_bc_sparse1.npy",
            # path="demos/icml2020/hand/door_bc_sparse3.npy",
            path="demos/icml2020/hand/door_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-sparse-v0': {
        'env_id': 'relocate-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/relocate2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/relocate_bc_sparse1.npy",
            path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'hammer-sparse-v0': {
        'env_id': 'hammer-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/hammer2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/hand/hammer_bc_sparse1.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
}

def compute_hand_sparse_reward(next_obs, reward, done, info):
    return info['goal_achieved'] - 1

def encoder_wrapped_env(variant):
    representation_size = 128
    output_classes = 20

    model_class = variant.get('model_class', TimestepPredictionModel)
    model = model_class(
        representation_size,
        # decoder_output_activation=decoder_activation,
        output_classes=output_classes,
        **variant['model_kwargs'],
    )
    # model = torch.nn.DataParallel(model)

    model_path = variant.get("model_path")
    # model = load_local_or_remote_file(model_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(ptu.device)
    model.eval()

    traj = np.load(variant.get("desired_trajectory"), allow_pickle=True)[0]

    goal_image = traj["observations"][-1]["image_observation"]
    goal_image = goal_image.reshape(1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
    # goal_image = goal_image.reshape(1, 300, 500, 3).transpose([0, 3, 1, 2]) / 255.0 # BECAUSE RLBENCH DEMOS ARENT IMAGE_ENV WRAPPED
    # goal_image = goal_image[:, :, :240, 60:500]
    goal_image = goal_image[:, :, 60:, 60:500]
    goal_image_pt = ptu.from_numpy(goal_image)
    save_image(goal_image_pt.data.cpu(), 'gitignore/goal.png', nrow=1)
    goal_latent = model.encode(goal_image_pt).detach().cpu().numpy().flatten()

    initial_image = traj["observations"][0]["image_observation"]
    initial_image = initial_image.reshape(1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
    # initial_image = initial_image.reshape(1, 300, 500, 3).transpose([0, 3, 1, 2]) / 255.0
    # initial_image = initial_image[:, :, :240, 60:500]
    initial_image = initial_image[:, :, 60:, 60:500]
    initial_image_pt = ptu.from_numpy(initial_image)
    save_image(initial_image_pt.data.cpu(), 'gitignore/initial.png', nrow=1)
    initial_latent = model.encode(initial_image_pt).detach().cpu().numpy().flatten()

    # Move these to td3_bc and bc_v3 (or at least type for reward_params)
    reward_params = dict(
        goal_latent=goal_latent,
        initial_latent=initial_latent,
        type=variant["reward_params_type"],
    )

    config_params = variant.get("config_params")

    env = variant['env_class'](**variant['env_kwargs'])
    env = ImageEnv(env,
        recompute_reward=False,
        transpose=True,
        image_length=450000,
        reward_type="image_distance",
        # init_camera=sawyer_pusher_camera_upright_v2,
    )
    env = EncoderWrappedEnv(
        env,
        model,
        reward_params,
        config_params,
        **variant.get("encoder_wrapped_env_kwargs", dict())
    )
    env = FlatGoalEnv(env, obs_keys=["state_observation", ])

    return env


def resume(variant):
    data = load_local_or_remote_file(variant.get("pretrained_algorithm_path"), map_location="cuda")
    algo = data['algorithm']

    algo.num_epochs = variant['num_epochs']

    post_pretrain_hyperparams = variant["trainer_kwargs"].get("post_pretrain_hyperparams", {})
    algo.trainer.set_algorithm_weights(**post_pretrain_hyperparams)

    algo.train()


def process_args(variant):
    if variant.get("debug", False):
        variant['max_path_length'] = 50
        variant['batch_size'] = 5
        variant['num_epochs'] = 5
        variant['num_eval_steps_per_epoch'] = 100
        variant['num_expl_steps_per_train_loop'] = 100
        variant['num_trains_per_train_loop'] = 10
        variant['min_num_steps_before_training'] = 100
        variant['trainer_kwargs']['num_pretrain_steps'] = min(10, variant['trainer_kwargs'].get('num_pretrain_steps', 0))


def experiment(variant):
    import mj_envs

    expl_env = gym.make(variant['env'])
    eval_env = gym.make(variant['env'])

    action_dim = int(np.prod(eval_env.action_space.shape))
    state_dim = obs_dim = np.prod(expl_env.observation_space.shape)
    M = 256

    qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
    qf_kwargs['output_size'] = 1
    qf_kwargs['input_size'] = action_dim + state_dim
    qf1 = MlpQf(**qf_kwargs)
    qf2 = MlpQf(**qf_kwargs)

    target_qf_kwargs = copy.deepcopy(qf_kwargs)
    target_qf1 = MlpQf(**target_qf_kwargs)
    target_qf2 = MlpQf(**target_qf_kwargs)

    policy_kwargs = copy.deepcopy(variant['policy_kwargs'])
    policy_kwargs['action_dim'] = action_dim
    policy_kwargs['obs_dim'] = state_dim
    policy = TanhGaussianPolicy(**policy_kwargs)

    vae_policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        latent_dim=action_dim * 2,
    )

    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        **variant['expl_path_collector_kwargs']
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        # save_images=False,
        MakeDeterministic(policy),
        **variant['eval_path_collector_kwargs']
    )

    # vae_eval_path_collector = MdpPathCollector(
    #     eval_env,
    #     vae_policy,
    #     # max_num_epoch_paths_saved=5,
    #     # save_images=False,
    # )


    replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )
    demo_train_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )
    demo_test_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )


    trainer = BEARTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vae=vae_policy,
        replay_buffer=replay_buffer,
        **variant['trainer_kwargs']
    )

    path_loader_class = variant.get('path_loader_class', MDPPathLoader)
    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    path_loader = path_loader_class(trainer,
                                    replay_buffer=replay_buffer,
                                    demo_train_buffer=demo_train_replay_buffer,
                                    demo_test_buffer=demo_test_replay_buffer,
                                    **path_loader_kwargs,
                                    # demo_off_policy_path=variant['data_path'],
                                    )
    # path_loader.load_bear_demos(pickled=False)
    path_loader.load_demos()
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        # vae_evaluation_data_collector=vae_eval_path_collector,
        replay_buffer=replay_buffer,
        # q_learning_alg=True,
        # batch_rl=variant['batch_rl'],
        **variant['algo_kwargs']
    )


    algorithm.to(ptu.device)
    trainer.pretrain_q_with_bc_data(256)
    algorithm.train()
