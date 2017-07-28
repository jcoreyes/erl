import argparse
import pickle
import random

import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    StateDistanceQLearning,
)
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.reacher_env import (
    FullStateVaryingWeightReacherEnv,
)
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.policies.zero_policy import ZeroPolicy
from railrl.qfunctions.state_distance.outer_product_qfunction import \
    OuterProductQFunction
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.samplers.path_sampler import MultitaskPathSampler


def main(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    if variant['generate_data']:
        action_space = convert_gym_space(env.action_space)
        es = GaussianStrategy(
            action_space=action_space,
            max_sigma=0.2,
            min_sigma=0.2,
        )
        exploration_policy = ZeroPolicy(
            int(action_space.flat_dim),
        )
        sampler_params = variant['sampler_params']
        replay_buffer_size = (
            sampler_params['min_num_steps_to_collect']
            + sampler_params['max_path_length']
        )
        replay_buffer = SplitReplayBuffer(
            EnvReplayBuffer(
                replay_buffer_size,
                env,
                flatten=True,
            ),
            EnvReplayBuffer(
                replay_buffer_size,
                env,
                flatten=True,
            ),
            fraction_paths_in_train=0.8,
        )
        sampler = MultitaskPathSampler(
            env,
            exploration_strategy=es,
            exploration_policy=exploration_policy,
            replay_buffer=replay_buffer,
            **variant['sampler_params']
        )
        sampler.collect_data()
        replay_buffer = sampler.replay_buffer
    else:
        dataset_path = variant['dataset_path']
        with open(dataset_path, 'rb') as handle:
            replay_buffer = pickle.load(handle)

    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    qf = variant['qf_class'](
        int(observation_space.flat_dim) + env.goal_dim,
        int(action_space.flat_dim),
        32,
        32,
        batchnorm_obs=False,
    )
    policy = FeedForwardPolicy(
        int(observation_space.flat_dim) + env.goal_dim,
        int(action_space.flat_dim),
        400,
        300,
    )
    algo = StateDistanceQLearning(
        env,
        qf,
        policy,
        replay_buffer=replay_buffer,
        exploration_policy=None,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_path', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    n_seeds = 1
    mode = "here"
    use_gpu = True
    exp_prefix = "7-27-full-state-vary-reward-weight-outer-prod-qf"
    snapshot_mode = 'gap'
    snapshot_gap = 5

    if mode == 'ec2':
        use_gpu = False

    dataset_path = args.replay_path

    # noinspection PyTypeChecker
    variant = dict(
        dataset_path=str(dataset_path),
        algo_params=dict(
            num_batches=100000,
            num_batches_per_epoch=1000,
            use_soft_update=True,
            tau=1e-3,
            batch_size=1024,
            discount=0.,
            qf_learning_rate=1e-4,
            policy_learning_rate=1e-5,
            sample_goals_from='replay_buffer',
            num_epochs=101,
        ),
        # env_class=GoalStateSimpleStateReacherEnv,
        # env_class=XyMultitaskSimpleStateReacherEnv,
        env_class=FullStateVaryingWeightReacherEnv,
        env_params=dict(
            add_noop_action=False,
            # reward_weights=[1, 1, 1, 1, 0, 0],
        ),
        sampler_params=dict(
            min_num_steps_to_collect=10000,
            max_path_length=1000,
            render=False,
        ),
        generate_data=args.replay_path is None,
        qf_class=OuterProductQFunction,
    )

    seed = random.randint(0, 10000)
    run_experiment(
        main,
        exp_prefix=exp_prefix,
        seed=seed,
        mode=mode,
        variant=variant,
        exp_id=0,
        use_gpu=use_gpu,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
    )
