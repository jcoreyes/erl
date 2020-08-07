from rlkit.envs.multitask.multitask_env import MultitaskToFlatEnv
from rlkit.envs.multitask.point2d import MultitaskImagePoint2DEnv
from rlkit.envs.mujoco.pusher2d import Pusher2DEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.td3.td3 import TD3
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.envs.vae_wrappers import VAEWrappedImageGoalEnv
import torch
from rlkit.envs.multitask.pusher2d import FullPusher2DEnv
from rlkit.envs.wrappers import ImageMujocoEnv

def experiment(variant):
    rdim = variant["rdim"]
    vae_paths = {
        2: "/home/ashvin/data/s3doodad/ashvin/vae/new-reacher2d-random/run0/id0/params.pkl",
        4: "/home/ashvin/data/s3doodad/ashvin/vae/new-reacher2d-random/run0/id1/params.pkl",
        8: "/home/ashvin/data/s3doodad/ashvin/vae/new-reacher2d-random/run0/id2/params.pkl",
        16: "/home/ashvin/data/s3doodad/ashvin/vae/new-reacher2d-random/run0/id3/params.pkl"
    }
    vae_path = vae_paths[rdim]
    vae = torch.load(vae_path)
    print("loaded", vae_path)

    if variant['multitask']:
        env = FullPusher2DEnv(**variant["env_kwargs"])
        env = ImageMujocoEnv(env, 84, camera_name="topview", transpose=True, normalize=True)
        env = VAEWrappedImageGoalEnv(env, vae, use_vae_obs=True, use_vae_reward=True, use_vae_goals=True,
            render_goals=True, render_rollouts=True, track_qpos_goal=5)
        env = MultitaskToFlatEnv(env)
    # else:
        # env = Pusher2DEnv(**variant['env_kwargs'])
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(action_space=env.action_space)
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=0.1,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = ConcatMlp(
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
    algorithm = TD3(
        env,
        training_env=env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    print("use_gpu", variant["use_gpu"], bool(variant["use_gpu"]))
    if variant["use_gpu"]:
        gpu_id = variant["gpu_id"]
        ptu.set_gpu_mode(True)
        ptu.set_device(gpu_id)
        algorithm.to(ptu.device)
        env._wrapped_env.vae.to(ptu.device)
    algorithm.train()


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
            ignore_multitask_goal=True,
            include_puck=False,
            arm_range=2,
        ),
        algorithm='TD3',
        multitask=True,
        normalize=False,
        rdim=4,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'pusher-2d-state-baselines-h100-multitask-less-shaped'

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.reward_scale': [0.1],
        'rdim': [2],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=2)
