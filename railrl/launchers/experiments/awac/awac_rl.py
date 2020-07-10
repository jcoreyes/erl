import gym
# import roboverse
from railrl.data_management.awr_env_replay_buffer import AWREnvReplayBuffer
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.wrappers import NormalizedBoxEnv, StackObservationEnv, RewardWrapperEnv
import railrl.torch.pytorch_util as ptu
from railrl.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from railrl.samplers.data_collector.step_collector import MdpStepCollector
from railrl.torch.networks import ConcatMlp
from railrl.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from railrl.torch.sac.awac_trainer import AWACTrainer
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
from railrl.envs.encoder_wrappers import VQVAEWrappedEnv
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

from railrl.envs.images import Renderer, InsertImageEnv, EnvRenderer

ENV_PARAMS = {

    'half-cheetah': {  # 6 DoF
        'env_id':'HalfCheetah-v2',
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'HalfCheetah-v2',
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/hc_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/hc_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
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
        'env_id':'Ant-v2',
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/ant_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/ant_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'walker': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Walker2d-v2',
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/walker_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/walker_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
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
    'SawyerRigGrasp-v0': {
        'env_id': 'SawyerRigGrasp-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 50,
        # 'num_epochs': 1000,
    },

    'pen-notermination-v0': {
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

    'pen-binary-v0': {
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            # path="demos/icml2020/hand/pen2_sparse.npy",
            path="demos/icml2020/hand/sparsity/railrl_pen-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/pen_bc_sparse1.npy",
            # path="demos/icml2020/hand/pen_bc_sparse2.npy",
            # path="demos/icml2020/hand/pen_bc_sparse3.npy",
            # path="demos/icml2020/hand/pen_bc_sparse4.npy",
            # path="demos/icml2020/hand/sparsity/bc/pen_bc_sparse4.npy",
            path="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10/id*/video_*_*.p",
            sync_dir="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'pen-binary-old-v0': {
        'env_id': 'pen-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/pen2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_pen-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/pen_bc_sparse1.npy",
            # path="demos/icml2020/hand/pen_bc_sparse2.npy",
            # path="demos/icml2020/hand/pen_bc_sparse3.npy",
            # path="demos/icml2020/hand/pen_bc_sparse4.npy",
            path="demos/icml2020/hand/pen_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'pen-sparse-v0': {
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/sparsity/railrl_pen-sparse-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/pen_bc_sparse1.npy",
            # path="demos/icml2020/hand/pen_bc_sparse2.npy",
            # path="demos/icml2020/hand/pen_bc_sparse3.npy",
            # path="demos/icml2020/hand/pen_bc_sparse4.npy",
            path="ashvin/icml2020/hand/sparsity/bc/pen-sparse1/run10/id*/video_*_*.p",
            sync_dir="ashvin/icml2020/hand/sparsity/bc/pen-sparse1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-binary-v0': {
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            # path="demos/icml2020/hand/door2_sparse.npy",
            path="demos/icml2020/hand/sparsity/railrl_door-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/door_bc_sparse1.npy",
            # path="demos/icml2020/hand/door_bc_sparse3.npy",
            # path="demos/icml2020/hand/door_bc_sparse4.npy",
            path="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10/id*/video_*_*.p",
            sync_dir="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-binary-old-v0': {
        'env_id': 'door-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/door2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_door-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/door_bc_sparse1.npy",
            # path="demos/icml2020/hand/door_bc_sparse3.npy",
            path="demos/icml2020/hand/door_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-sparse-v0': {
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/sparsity/railrl_door-sparse-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/door_bc_sparse1.npy",
            # path="demos/icml2020/hand/door_bc_sparse3.npy",
            # path="demos/icml2020/hand/door_bc_sparse4.npy",
            path="ashvin/icml2020/hand/sparsity/bc/door-sparse1/run10/id*/video_*_*.p",
            sync_dir="ashvin/icml2020/hand/sparsity/bc/door-sparse1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-binary-v0': {
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            # path="demos/icml2020/hand/relocate2_sparse.npy",
            path="demos/icml2020/hand/sparsity/railrl_relocate-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/relocate_bc_sparse1.npy",
            # path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            path="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10/id*/video_*_*.p",
            sync_dir="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-binary-old-v0': {
        'env_id': 'relocate-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/relocate2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_relocate-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/relocate_bc_sparse1.npy",
            path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-sparse-v0': {
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/sparsity/railrl_relocate-sparse-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/relocate_bc_sparse1.npy",
            # path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            path="ashvin/icml2020/hand/sparsity/bc/relocate-sparse1/run10/id*/video_*_*.p",
            sync_dir="ashvin/icml2020/hand/sparsity/bc/relocate-sparse1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },

    'hammer-binary-v0': {
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
    'hammer-sparse-v0': {
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
        variant['trainer_kwargs']['bc_num_pretrain_steps'] = min(10, variant['trainer_kwargs'].get('bc_num_pretrain_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain1_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain1_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain2_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain2_steps', 0))

def experiment(variant):

    # if 'env' in variant:
    if variant.get("pretrained_algorithm_path", False):
        resume(variant)
        return
    env_params = ENV_PARAMS.get(variant.get('env'), {})
    variant.update(env_params)
    env_name = variant.get("env", None)
    env_id = variant.get('env_id', None)
    env_class = variant.get('env_class', None)

    if env_name in [
        'pen-v0', 'pen-sparse-v0', 'pen-notermination-v0', 'pen-binary-v0', 'pen-binary-old-v0',
        'door-v0', 'door-sparse-v0', 'door-binary-v0', 'door-binary-old-v0',
        'relocate-v0', 'relocate-sparse-v0', 'relocate-binary-v0', 'relocate-binary-old-v0',
        'hammer-v0', 'hammer-sparse-v0', 'hammer-binary-v0',
    ]:
        import mj_envs
        expl_env = gym.make(env_params.get('env_id', env_name))
        eval_env = gym.make(env_params.get('env_id', env_name))
    elif env_name in [ # D4RL envs
        "maze2d-open-v0", "maze2d-umaze-v0", "maze2d-medium-v0", "maze2d-large-v0",
        "maze2d-open-dense-v0", "maze2d-umaze-dense-v0", "maze2d-medium-dense-v0", "maze2d-large-dense-v0",
        "antmaze-umaze-v0", "antmaze-umaze-diverse-v0", "antmaze-medium-diverse-v0",
        "antmaze-medium-play-v0", "antmaze-large-diverse-v0", "antmaze-large-play-v0",
        "pen-human-v0", "pen-cloned-v0", "pen-expert-v0", "hammer-human-v0", "hammer-cloned-v0", "hammer-expert-v0",
        "door-human-v0", "door-cloned-v0", "door-expert-v0", "relocate-human-v0", "relocate-cloned-v0", "relocate-expert-v0",
        "halfcheetah-random-v0", "halfcheetah-medium-v0", "halfcheetah-expert-v0", "halfcheetah-mixed-v0", "halfcheetah-medium-expert-v0",
        "walker2d-random-v0", "walker2d-medium-v0", "walker2d-expert-v0", "walker2d-mixed-v0", "walker2d-medium-expert-v0",
        "hopper-random-v0", "hopper-medium-v0", "hopper-expert-v0", "hopper-mixed-v0", "hopper-medium-expert-v0"
    ]:
        import d4rl
        expl_env = gym.make(env_name)
        eval_env = gym.make(env_name)
    elif env_id:
        expl_env = NormalizedBoxEnv(gym.make(env_id))
        eval_env = NormalizedBoxEnv(gym.make(env_id))

        # print("CHANGE BUFFER TO 1 MIL AGAIN!!!")
        # from railrl.envs.images import InsertImageEnv, EnvRenderer
        # renderer = EnvRenderer(init_camera=None, **{})
        # expl_env = InsertImageEnv(gym.make(env_id), renderer=renderer)
        # eval_env = InsertImageEnv(gym.make(env_id), renderer=renderer)

        # expl_env = VQVAEWrappedEnv(
        #     expl_env,
        #     model,
        #     dict(image_observation="latent_observation",),)

        # eval_env = VQVAEWrappedEnv(
        #     eval_env,
        #     model,
        #     dict(image_observation="latent_observation",),)

        # expl_env = NormalizedBoxEnv(expl_env)
        # eval_env = NormalizedBoxEnv(eval_env)
    elif env_class:
        env_kwargs = variant.get("env_kwargs", {})
        expl_env = NormalizedBoxEnv(env_class(**env_kwargs))
        eval_env = NormalizedBoxEnv(env_class(**env_kwargs))
    else:
        expl_env = NormalizedBoxEnv(variant['env']())
        eval_env = NormalizedBoxEnv(variant['env']())

        if 'env_id' in env_params:
            if env_params['env_id'] in ['pen-v0', 'pen-sparse-v0', 'door-v0', 'relocate-v0', 'hammer-v0',
                                        'pen-sparse-v0', 'door-sparse-v0', 'relocate-sparse-v0', 'hammer-sparse-v0']:
                import mj_envs
            # if env_params['env_id'] in ['SawyerRigGrasp-v0']:
            #     expl_env = InsertImageEnv(roboverse.make(env_params['env_id']))
            #     eval_env = InsertImageEnv(roboverse.make(env_params['env_id']))
            #     expl_env = EncoderWrappedEnv(
            #         expl_env,
            #         model,
            #         dict(image_observation="latent_observation",),)

            #     eval_env = EncoderWrappedEnv(
            #         eval_env,
            #         model,
            #         dict(image_observation="latent_observation",),)

            else:
                expl_env = gym.make(env_params['env_id'])
                eval_env = gym.make(env_params['env_id'])
        else:
            expl_env = NormalizedBoxEnv(variant['env_class']())
            eval_env = NormalizedBoxEnv(variant['env_class']())

    # if variant.get('sparse_reward', False):
    #     expl_env = RewardWrapperEnv(expl_env, compute_hand_sparse_reward)
    #     eval_env = RewardWrapperEnv(eval_env, compute_hand_sparse_reward)

    if variant.get('add_env_demos', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])
    if variant.get('add_env_offpolicy_data', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])

    # else:
    #     expl_env = encoder_wrapped_env(variant)
    #     eval_env = encoder_wrapped_env(variant)

    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    stack_obs = path_loader_kwargs.get("stack_obs", 1)
    if stack_obs > 1:
        expl_env = StackObservationEnv(expl_env, stack_obs=stack_obs)
        eval_env = StackObservationEnv(eval_env, stack_obs=stack_obs)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    if hasattr(expl_env, 'info_sizes'):
        env_info_sizes = expl_env.info_sizes
    else:
        env_info_sizes = dict()

    qf_kwargs = variant.get("qf_kwargs", {})
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )

    policy_class = variant.get("policy_class", TanhGaussianPolicy)
    policy_kwargs = variant['policy_kwargs']
    policy_path = variant.get("policy_path", False)
    if policy_path:
        policy = load_local_or_remote_file(policy_path)
    else:
        policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        )
    buffer_policy_path = variant.get("buffer_policy_path", False)
    if buffer_policy_path:
        buffer_policy = load_local_or_remote_file(buffer_policy_path)
    else:
        buffer_policy_class = variant.get("buffer_policy_class", policy_class)
        buffer_policy = buffer_policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant.get("buffer_policy_kwargs", policy_kwargs),
        )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )

    expl_policy = policy
    exploration_kwargs =  variant.get('exploration_kwargs', {})
    if exploration_kwargs:
        if exploration_kwargs.get("deterministic_exploration", False):
            expl_policy = MakeDeterministic(policy)

        exploration_strategy = exploration_kwargs.get("strategy", None)
        if exploration_strategy is None:
            pass
        elif exploration_strategy == 'ou':
            es = OUStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        elif exploration_strategy == 'gauss_eps':
            es = GaussianAndEpislonStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],  # constant sigma
                epsilon=0,
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        else:
            error

    if variant.get('replay_buffer_class', EnvReplayBuffer) == AWREnvReplayBuffer:
        main_replay_buffer_kwargs = variant['replay_buffer_kwargs']
        main_replay_buffer_kwargs['env'] = expl_env
        main_replay_buffer_kwargs['qf1'] = qf1
        main_replay_buffer_kwargs['qf2'] = qf2
        main_replay_buffer_kwargs['policy'] = policy
    else:
        main_replay_buffer_kwargs=dict(
            max_replay_buffer_size=variant['replay_buffer_size'],
            env=expl_env,
        )
    replay_buffer_kwargs = dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )

    replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        **main_replay_buffer_kwargs,
    )
    if variant.get('use_validation_buffer', False):
        train_replay_buffer = replay_buffer
        validation_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
            **main_replay_buffer_kwargs,
        )
        replay_buffer = SplitReplayBuffer(train_replay_buffer, validation_replay_buffer, 0.9)

    trainer_class = variant.get("trainer_class", AWACTrainer)
    trainer = trainer_class(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        buffer_policy=buffer_policy,
        **variant['trainer_kwargs']
    )
    if variant['collection_mode'] == 'online':
        expl_path_collector = MdpStepCollector(
            expl_env,
            policy,
        )
        algorithm = TorchOnlineRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=variant['max_path_length'],
            batch_size=variant['batch_size'],
            num_epochs=variant['num_epochs'],
            num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
            num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
            num_trains_per_train_loop=variant['num_trains_per_train_loop'],
            min_num_steps_before_training=variant['min_num_steps_before_training'],
        )
    else:
        expl_path_collector = MdpPathCollector(
            expl_env,
            expl_policy,
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=variant['max_path_length'],
            batch_size=variant['batch_size'],
            num_epochs=variant['num_epochs'],
            num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
            num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
            num_trains_per_train_loop=variant['num_trains_per_train_loop'],
            min_num_steps_before_training=variant['min_num_steps_before_training'],
        )
    algorithm.to(ptu.device)

    demo_train_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )
    demo_test_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )

    if variant.get("save_video", False):
        if variant.get("presampled_goals", None):
            variant['image_env_kwargs']['presampled_goals'] = load_local_or_remote_file(variant['presampled_goals']).item()

        def get_img_env(env):
            renderer = EnvRenderer(**variant["renderer_kwargs"])
            img_env = InsertImageEnv(GymToMultiEnv(env), renderer=renderer)

        image_eval_env = ImageEnv(GymToMultiEnv(eval_env), **variant["image_env_kwargs"])
        # image_eval_env = get_img_env(eval_env)
        image_eval_path_collector = ObsDictPathCollector(
            image_eval_env,
            eval_policy,
            observation_key="state_observation",
        )
        image_expl_env = ImageEnv(GymToMultiEnv(expl_env), **variant["image_env_kwargs"])
        # image_expl_env = get_img_env(expl_env)
        image_expl_path_collector = ObsDictPathCollector(
            image_expl_env,
            expl_policy,
            observation_key="state_observation",
        )
        video_func = VideoSaveFunction(
            image_eval_env,
            variant,
            image_expl_path_collector,
            image_eval_path_collector,
        )
        algorithm.post_train_funcs.append(video_func)
    if variant.get('save_paths', False):
        algorithm.post_train_funcs.append(save_paths)
    if variant.get('load_demos', False):
        path_loader_class = variant.get('path_loader_class', MDPPathLoader)
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos()
    if variant.get('load_env_dataset_demos', False):
        path_loader_class = variant.get('path_loader_class', HDF5PathLoader)
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos(expl_env.get_dataset())
    if variant.get('save_initial_buffers', False):
        buffers = dict(
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
        )
        buffer_path = osp.join(logger.get_snapshot_dir(), 'buffers.p')
        pickle.dump(buffers, open(buffer_path, "wb"))
    if variant.get('pretrain_buffer_policy', False):
        trainer.pretrain_policy_with_bc(
            buffer_policy,
            replay_buffer.train_replay_buffer,
            replay_buffer.validation_replay_buffer,
            10000,
            label="buffer",
        )
    if variant.get('pretrain_policy', False):
        trainer.pretrain_policy_with_bc(
            policy,
            demo_train_buffer,
            demo_test_buffer,
            trainer.bc_num_pretrain_steps,
        )
    if variant.get('pretrain_rl', False):
        trainer.pretrain_q_with_bc_data()
    if variant.get('save_pretrained_algorithm', False):
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        torch.save(data, open(pt_path, "wb"))
        torch.save(data, open(p_path, "wb"))
    if variant.get('train_rl', True):
        algorithm.train()
