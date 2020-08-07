"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
from rlkit.launchers.launcher_util import run_experiment
import rlkit.misc.hyperparameter as hyp

from rlkit.launchers.exp_launcher import rl_experiment

from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

if __name__ == "__main__":
    variant = dict(
        env_id='SawyerReachXYEnv-v1', #'SawyerPushAndReachEnvEasy-v0',
        init_camera=sawyer_xyz_reacher_camera_v0,
        rl_variant=dict(
            do_state_exp=True,
            algo_kwargs=dict(
                num_epochs=300,
                batch_size=128,
                num_eval_steps_per_epoch=1000,
                num_expl_steps_per_train_loop=1000,
                num_trains_per_train_loop=1000,
            ),
            max_path_length=100,
            td3_trainer_kwargs=dict(),
            twin_sac_trainer_kwargs=dict(),
            replay_buffer_kwargs=dict(
                max_size=int(1E6),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
            ),
            exploration_noise=0.1,
            exploration_type='epsilon',
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algorithm="SAC", #"SAC",

            dump_video_kwargs=dict(
                rows=1,
                columns=3,
            ),
            save_video_period=2,

            # do_state_exp=True,
            # algorithm='td3',
            # algo_kwargs=dict(
            #     num_epochs=100,
            #     max_path_length=50,
            #     batch_size=128,
            #     num_eval_steps_per_epoch=1000,
            #     num_expl_steps_per_train_loop=1000,
            #     num_trains_per_train_loop=1000,
            #     min_num_steps_before_training=10000,
            # ),
            # trainer_kwargs=dict(
            #     discount=0.99,
            # ),
            # replay_buffer_kwargs=dict(
            #     max_size=100000,
            #     fraction_goals_rollout_goals=0.2,
            #     fraction_goals_env_goals=0.0,
            # ),
        ),
    )
    # setup_logger('her-td3-sawyer-experiment', variant=variant)
    # experiment(variant)
    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                rl_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                time_in_mins=1000,
          )
