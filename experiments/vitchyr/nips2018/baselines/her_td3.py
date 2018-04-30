from gym.envs.robotics import FetchPushEnv

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer, \
    RelabelingReplayBuffer
from railrl.envs.mujoco.sawyer_gripper_env import SawyerPushXYEnv, SawyerXYZEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.her.her_td3 import HerTd3
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
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
    goal_dim = env.goal_dim
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
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=300,
            num_steps_per_epoch=10000,
            num_steps_per_eval=10000,
            # num_epochs=50,
            # num_steps_per_epoch=100,
            # num_steps_per_eval=100,
            max_path_length=100,
            num_updates_per_env_step=1,
            batch_size=100,
            discount=0.99,
        ),
        env_class=SawyerPushXYEnv,
        # env_class=SawyerXYZEnv,
        # env_class=FetchPushEnv,
        env_kwargs=dict(
            frame_skip=50,
            pos_action_scale=2./100,
        ),
        replay_buffer_class=RelabelingReplayBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
        ),
        normalize=True,
        algorithm='HER-TD3',
        version='her',
    )
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer-push-longer-frameskip-and-pos-scale-2'

    search_space = {
        # 'env_kwargs.randomize_goals': [True, False],
        # 'env_kwargs.only_reward_block_to_goal': [False, True],
        # 'replay_buffer_kwargs.num_goals_to_sample': [4],
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals': [0.2],
        'algo_kwargs.num_updates_per_env_step': [
            1,
            # 5,
        ],
        'algo_kwargs.max_path_length': [
            100,
            50,
        ],
        'exploration_type': [
            'epsilon',
            'ou',
            'gaussian',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
            )
