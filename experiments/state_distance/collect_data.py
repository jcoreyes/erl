from railrl.algos.state_distance.util import get_replay_buffer
from railrl.envs.multitask.reacher_env import (
    XyMultitaskSimpleStateReacherEnv,
)
from railrl.launchers.launcher_util import run_experiment


def main(variant):
    get_replay_buffer(variant, save_replay_buffer=True)


if __name__ == '__main__':
    # out_dir = Path(LOG_DIR) / 'datasets/generated'
    # out_dir /= '7-25--xy-multitask-simple-state--100k--add-no-op'
    # out_dir = str(out_dir)
    out_dir = None
    min_num_steps_to_collect = 10000
    max_path_length = 1000
    replay_buffer_size = min_num_steps_to_collect + max_path_length

    # noinspection PyTypeChecker
    variant = dict(
        out_dir=str(out_dir),
        sampler_params=dict(
            min_num_steps_to_collect=min_num_steps_to_collect,
            max_path_length=max_path_length,
            render=True,
        ),
        env_class=XyMultitaskSimpleStateReacherEnv,
        env_params=dict(
            add_noop_action=False,
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
        base_log_dir=out_dir,
    )
