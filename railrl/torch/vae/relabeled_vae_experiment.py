from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv, MultitaskEnvToSilentMultitaskEnv
from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
from railrl.envs.mujoco.pusher2d import Pusher2DEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
import railrl.torch.pytorch_util as ptu
from railrl.torch.td3.td3 import TD3
import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants

from railrl.envs.wrappers import ImageMujocoEnv
from railrl.envs.vae_wrappers import VAEWrappedImageGoalEnv, VAEWrappedEnv
from railrl.data_management.her_replay_buffer import RelabelingReplayBuffer
from railrl.torch.her.relabeled_td3 import RelabeledTd3
import torch

from railrl.core import logger
import os.path as osp

def experiment(variant):
    env = variant["env"](**variant['env_kwargs'])

    do_state_based_exp = variant.get("do_state_based_exp", False)

    if not do_state_based_exp:
        rdim = variant["rdim"]
        use_env_goals = variant["use_env_goals"]
        vae_path = variant["vae_paths"][str(rdim)]
        render = variant["render"]
        wrap_mujoco_env = variant.get("wrap_mujoco_env", False)
        reward_params = variant.get("reward_params", dict())

        if wrap_mujoco_env:
            env = ImageMujocoEnv(env, 84, camera_name="topview", transpose=True, normalize=True)


        if use_env_goals:
            track_qpos_goal = variant.get("track_qpos_goal", 0)
            env = VAEWrappedImageGoalEnv(env, vae_path, use_vae_obs=True,
                use_vae_reward=True, use_vae_goals=True,
                render_goals=render, render_rollouts=render,
                track_qpos_goal=track_qpos_goal, reward_params=reward_params)
        else:
            env = VAEWrappedEnv(env, vae_path, use_vae_obs=True,
                use_vae_reward=True, use_vae_goals=True,
                render_goals=render, render_rollouts=render,
                reward_params=reward_params)

        vae_wrapped_env = env

    # import ipdb; ipdb.set_trace()
    env = MultitaskEnvToSilentMultitaskEnv(env)
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    exploration_type = variant['exploration_type']
    exploration_noise = variant.get('exploration_noise', 0.1)
    if exploration_type == 'ou':
        es = OUStrategy(action_space=env.action_space)
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=exploration_noise,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    obs_dim = env.observation_space.low.size + env.goal_space.low.size
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = RelabelingReplayBuffer(
        max_size=100000,
        env=env,
        **variant['replay_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer
    algorithm = RelabeledTd3(
        env,
        training_env=env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        render=do_state_based_exp and variant.get("render", False),
        **variant['algo_kwargs']
    )

    print("use_gpu", variant["use_gpu"], bool(variant["use_gpu"]))
    if variant["use_gpu"]: # change this to standardized format
        gpu_id = variant["gpu_id"]
        ptu.set_gpu_mode(True)
        ptu.set_device(gpu_id)
        algorithm.cuda()
        if not do_state_based_exp:
            env._wrapped_env.vae.cuda()

    save_video = variant.get("save_video", True)
    if not do_state_based_exp and save_video:
        from railrl.torch.vae.sim_vae_policy import dump_video
        video_env = MultitaskToFlatEnv(vae_wrapped_env)
        logdir = logger.get_snapshot_dir()
        filename = osp.join(logdir, 'video_0.mp4')
        dump_video(video_env, policy, filename)

    algorithm.train()

    if not do_state_based_exp and save_video:
        filename = osp.join(logdir, 'video_final.mp4')
        dump_video(video_env, policy, filename)


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            # qf_learning_rate=1e-3,
            # policy_learning_rate=1e-4,
        ),
        env_kwargs=dict(
            render_onscreen=False,
            render_size=84,
            ignore_multitask_goal=True,
            ball_radius=1,
        ),
        algorithm='TD3',
        multitask=True,
        normalize=False,
        rdim=4,
        render=False,
        save_video=True,
    )

    n_seeds = 3

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.reward_scale': [0.01, 0.1, 1],
        'rdim': [2, 4, 8, 16],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=0)
