import argparse

from railrl.envs.mujoco.sawyer_reach_env import (
    SawyerReachXYEnv,
)
from railrl.envs.mujoco.sawyer_push_env import (
    SawyerPushXYEasyEnv,
    SawyerMultiPushEnv
)
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer, \
    RelabelingReplayBuffer
from railrl.misc.variant_generator import VariantGenerator
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.her.her_td3 import HerTd3

COMMON_PARAMS = dict(
    num_epochs=300,
    num_steps_per_epoch=1000,
    num_steps_per_eval=1000, #check
    max_path_length=100, #check
    # min_num_steps_before_training=1000, #check
    batch_size=128,
    discount=0.99,
    # soft_target_tau=1e-2,
    # target_update_period=1,  #check
    algorithm="HER-TD3",
    version="normal",
    env_class=SawyerReachXYEnv,
    normalize=True,
    replay_buffer_class=RelabelingReplayBuffer,
    replay_buffer_kwargs=dict(
        max_size=int(1E6),
        fraction_goals_are_rollout_goals=0.2,
        fraction_resampled_goals_are_env_goals=0.5,
    ),
    exploration_type=['epsilon', 'ou', 'gaussian'],
)

ENV_PARAMS = {
    'sawyer-reach-xy': { # 6 DoF
        'env_class': SawyerReachXYEnv,
        'num_epochs': 75,
        'reward_scale': [1e3, 1e4, 1e5] #[0.01, 0.1, 1, 10, 100],
    },
    'sawyer-push-xy-easy': {  # 6 DoF
        'env_class': SawyerPushXYEasyEnv,
        'num_epochs': 300,
        'reward_scale': [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]  # [0.01, 0.1, 1, 10, 100],
    },
    'sawyer-multi-push': {  # 6 DoF
        'env_class': SawyerMultiPushEnv,
        'num_epochs': 300,
        'reward_scale': [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]  # [0.01, 0.1, 1, 10, 100],
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default='sawyer-reach-xy')
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
    if variant['normalize']:
        env = NormalizedBoxEnv(variant['env_class']())
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    goal_dim = env.goal_dim
    variant['algo_kwargs'] = dict(
        num_epochs=variant['num_epochs'],
        num_steps_per_epoch=variant['num_steps_per_epoch'],
        num_steps_per_eval=variant['num_steps_per_eval'],
        max_path_length=variant['max_path_length'],
        # min_num_steps_before_training=variant['min_num_steps_before_training'],
        batch_size=variant['batch_size'],
        discount=variant['discount'],
        # soft_target_tau=variant['soft_target_tau'],
        # target_update_period=variant['target_update_period'],
        reward_scale=variant['reward_scale'],
    )

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
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
    replay_buffer = variant['replay_buffer_class'](
        env=env,
        **variant['replay_buffer_kwargs']
    )
    algorithm = HerTd3(
        env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        qf1.cuda()
        qf2.cuda()
        policy.cuda()
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = parse_args()
    variant_generator = get_variants(args)
    variants = variant_generator.variants()
    exp_prefix = "her-td3-" + args.env
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
