import argparse

from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    HopperEnv,
    Walker2dEnv,
    InvertedDoublePendulumEnv,
)
from gym.envs.classic_control import PendulumEnv

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.pytorch_util as ptu
from rlkit.misc.variant_generator import VariantGenerator
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.td3.td3 import TD3

COMMON_PARAMS = dict(
    num_epochs=3000,
    num_steps_per_epoch=1000,
    num_steps_per_eval=1000, #check
    max_path_length=1000, #check
    min_num_steps_before_training=1000, #check
    batch_size=256,
    discount=0.99,
    replay_buffer_size=int(1E6),
    layer_size=256, # [256, 512]
    algorithm="TD3",
    version="normal",
    env_class=HalfCheetahEnv,
    exploration_type=['epsilon', 'ou', 'gaussian'],
)

ENV_PARAMS = {
    'half-cheetah': { # 6 DoF
        'env_class': HalfCheetahEnv,
        'num_epochs': 3000, #4000
        'reward_scale': [10], #[0.1, 1, 100], # [0.1, 1, 3, 5, 10, 100], #[3,5]
    },
    'inv-double-pendulum': {  # 2 DoF
        'env_class': InvertedDoublePendulumEnv,
        'num_epochs': 50, #50
        'reward_scale': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], # [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    'ant': {  # 6 DoF
        'env_class': AntEnv,
        'num_epochs': 3000,  # 4000
        'reward_scale': [10], #[0.1, 1, 100], # [0.1, 1, 5, 10, 100],  # [5,10],
    },
    'walker': {  # 6 DoF
        'env_class': Walker2dEnv,
        'num_epochs': 3000,  # 4000
        'reward_scale': [0.1, 1, 10, 100] #[0.1, 1, 3, 5, 10, 100],  # [3,5,10],
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default='inv-double-pendulum')
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    return args

def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = COMMON_PARAMS
    params.update(env_params)

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg

def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    variant['algo_kwargs'] = dict(
        num_epochs=variant['num_epochs'],
        num_steps_per_epoch=variant['num_steps_per_epoch'],
        num_steps_per_eval=variant['num_steps_per_eval'],
        max_path_length=variant['max_path_length'],
        min_num_steps_before_training=variant['min_num_steps_before_training'],
        batch_size=variant['batch_size'],
        discount=variant['discount'],
        replay_buffer_size=variant['replay_buffer_size'],
        reward_scale=variant['reward_scale'],
    )

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
        # **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
        # **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
        # **variant['policy_kwargs']
    )
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
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TD3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        qf1.to(ptu.device)
        qf2.to(ptu.device)
        policy.to(ptu.device)
        algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = parse_args()
    variant_generator = get_variants(args)
    variants = variant_generator.variants()
    exp_prefix = "td3-" + args.env
    if len(args.label) > 0:
        exp_prefix = exp_prefix + "-" + args.label

    for _ in range(args.num_seeds):
        for exp_id, variant in enumerate(variants):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=args.mode,
                exp_id=exp_id,
                variant=variant,
                use_gpu=args.gpu,
            )
