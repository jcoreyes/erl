import argparse

from gym.envs.mujoco import PusherEnv

from rlkit.tf.state_distance.util import get_replay_buffer
from rlkit.envs.multitask.reacher_env import (
    XyMultitaskSimpleStateReacherEnv,
)
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment


def main(variant):
    get_replay_buffer(variant, save_replay_buffer=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    min_num_steps_to_collect = 10000
    max_path_length = 1000
    replay_buffer_size = min_num_steps_to_collect + max_path_length

    # noinspection PyTypeChecker
    variant = dict(
        sampler_params=dict(
            min_num_steps_to_collect=min_num_steps_to_collect,
            max_path_length=max_path_length,
            render=args.render,
        ),
        # env_class=XyMultitaskSimpleStateReacherEnv,
        env_class=PusherEnv,
        env_params=dict(
        ),
        # sampler_es_class=GaussianStrategy,
        sampler_es_class=OUStrategy,
        sampler_es_params=dict(
            max_sigma=1,
            min_sigma=1,
        ),
        generate_data=True,
        replay_buffer_size=replay_buffer_size,
    )
    # main(variant)
    run_experiment(
        main,
        exp_prefix='pusher-ou-sigma-0p3-100k',
        seed=0,
        mode='here',
        variant=variant,
        exp_id=0,
        use_gpu=True,
        snapshot_mode='last',
    )
