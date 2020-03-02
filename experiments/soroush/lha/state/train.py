from railrl.launchers.launcher_util import run_experiment
import railrl.misc.hyperparameter as hyp
import argparse
import math

from railrl.launchers.exp_launcher import rl_experiment

from multiworld.envs.mujoco.cameras import *

variant = dict(
    rl_variant=dict(
        do_state_exp=True,
        algo_kwargs=dict(
            num_epochs=1000,
            batch_size=128,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=4000,
            min_num_steps_before_training=1000,
        ),
        max_path_length=100,
        td3_trainer_kwargs=dict(),
        twin_sac_trainer_kwargs=dict(),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        exploration_type='gaussian_and_epsilon',
        es_kwargs=dict(
            max_sigma=.2,
            min_sigma=.2,
            epsilon=.3,
        ),
        algorithm="TD3",
        dump_video_kwargs=dict(
            rows=1,
            columns=8,
        ),
        save_video_period=50,
    ),
)

env_params = {
    'pnr-leap': {
        'env_id': [
            # 'SawyerPushDebugLEAP-v0',
            'SawyerPushDebugLEAP-v1',
            # 'SawyerPushDebugLEAP-v2',
        ],
        'init_camera':[sawyer_xyz_reacher_camera_v0],
    },
    'pnr-ccrig': {
        'env_id': [
            # 'SawyerPushDebugCCRIG-v0',
            'SawyerPushDebugCCRIG-v1',
            # 'SawyerPushDebugCCRIG-v2',
        ],
        'init_camera':[sawyer_xyz_reacher_camera_v0],
    },
}

def process_variant(variant):
    rl_variant = variant['rl_variant']

    mpl = rl_variant['max_path_length']
    rl_variant['td3_trainer_kwargs']['discount'] = 1 - 1 / mpl
    rl_variant['twin_sac_trainer_kwargs']['discount'] = 1 - 1 / mpl

    if args.debug:
        rl_variant['algo_kwargs']['num_epochs'] = 4
        rl_variant['algo_kwargs']['batch_size'] = 128
        rl_variant['replay_buffer_kwargs']['max_size'] = int(1e4)
        rl_variant['algo_kwargs']['num_eval_steps_per_epoch'] = 200
        rl_variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 200
        rl_variant['algo_kwargs']['num_trains_per_train_loop'] = 200
        rl_variant['dump_video_kwargs']['columns'] = 2
        rl_variant['save_video_period'] = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--max_exps_per_instance', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--first_variant_only', action='store_true')
    parser.add_argument('--no_video',  action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()

    if args.mode == 'local' and args.label == '':
        args.label = 'local'

    variant['exp_label'] = args.label

    search_space = env_params[args.env]
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    prefix_list = ['train', 'state', args.label]
    while None in prefix_list: prefix_list.remove(None)
    while '' in prefix_list: prefix_list.remove('')
    exp_prefix = '-'.join(prefix_list)

    if args.mode == 'ec2' and (not args.no_gpu):
        max_exps_per_instance = args.max_exps_per_instance
    else:
        max_exps_per_instance = 1

    num_exps_for_instances = np.ones(int(math.ceil(args.num_seeds / max_exps_per_instance)), dtype=np.int32) \
                             * max_exps_per_instance
    num_exps_for_instances[-1] -= (np.sum(num_exps_for_instances) - args.num_seeds)

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters(print_info=False)):
        process_variant(variant)
        for num_exps in num_exps_for_instances:
            run_experiment(
                rl_experiment,
                exp_prefix=exp_prefix,
                mode=args.mode,
                variant=variant,
                use_gpu=(not args.no_gpu),

                num_exps_per_instance=int(num_exps),

                snapshot_gap=50,
                snapshot_mode='gap_and_last',
          )
