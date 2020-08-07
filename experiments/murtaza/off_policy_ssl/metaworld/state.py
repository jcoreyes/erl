from rlkit.launchers.experiments.ashvin.multiworld import her_td3_experiment
import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open_6dof import SawyerDrawerOpen6DOFEnv
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=4000,
            min_num_steps_before_training=1000,
            max_path_length=100,
        ),
        trainer_kwargs=dict(),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='HER-TD3',
        version='normal',
        es_kwargs=dict(
            max_sigma=.2,
        ),
        exploration_type='ou',
        observation_key='state_observation',
        init_camera=sawyer_pusher_camera_upright_v2,
        do_state_exp=True,

        save_video=True,
        imsize=84,

        snapshot_mode='gap_and_last',
        snapshot_gap=50,

        env_class=SawyerDrawerOpen6DOFEnv,
        env_kwargs=dict(
            random_init=False,
        ),

        wrap_mujoco_gym_to_multi_env=False,
    )

    search_space = {
        'replay_buffer_kwargs.fraction_goals_rollout_goals': [1.0, ],
        'replay_buffer_kwargs.fraction_goals_env_goals': [0.0, ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local_docker'
    # exp_name = 'test'

    n_seeds = 2
    mode = 'gcp'
    exp_name = 'sawyer_drawer_open_ashvin_exp_v2'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                num_exps_per_instance=2,
                skip_wait=False,
                gcp_kwargs=dict(
                    preemptible=False,
                )
            )
