import argparse

from railrl.algos.state_distance.util import get_replay_buffer
from railrl.envs.multitask.reacher_env import (
    XyMultitaskSimpleStateReacherEnv,
)
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.launchers.launcher_util import run_experiment


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
        env_class=XyMultitaskSimpleStateReacherEnv,
        env_params=dict(
            add_noop_action=False,
        ),
        sampler_es_class=GaussianStrategy,
        sampler_es_params=dict(
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        generate_data=True,
        replay_buffer_size=replay_buffer_size,
    )
    # main(variant)
    run_experiment(
        main,
        exp_prefix='dev-uniform-10k',
        seed=0,
        mode='here',
        variant=variant,
        exp_id=0,
        use_gpu=True,
        snapshot_mode='last',
    )
