import rlkit.misc.hyperparameter as hyp
from rlkit.data_management.her_replay_buffer import RelabelingReplayBuffer
from rlkit.envs.mujoco.sawyer_reach_torque_env import SawyerReachTorqueEnv
from rlkit.launchers.experiments.vitchyr.multitask import her_td3_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=500,
            num_steps_per_epoch=50,
            num_steps_per_eval=1000,
            max_path_length=50,
            num_updates_per_env_step=1,
            batch_size=128,
            discount=0.99,
            min_num_steps_before_training=128,
        ),
        env_class=SawyerReachTorqueEnv,
        env_kwargs=dict(
            keep_vel_in_obs=False,
        ),
        replay_buffer_class=RelabelingReplayBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.1,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        normalize=True,
        algorithm='HER-TD3',
        version='normal',
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    search_space = {
        'algo_kwargs.num_updates_per_env_step': [
            1,
            5,
            10,
        ],
        'replay_buffer_kwargs.fraction_resampled_goals_are_env_goals': [
            0.5,
            1.0,
        ],
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals': [
            0.2,
        ],
        'env_kwargs.reward_info.type': [
            'euclidean',
        ],
        'exploration_type': [
            'ou',
        ],
        'history_len': [
            2,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if (
                variant['replay_buffer_kwargs']['fraction_goals_are_rollout_goals'] == 1.0
                and variant['replay_buffer_kwargs']['fraction_resampled_goals_are_env_goals'] != 0.0
        ):  # redundant setting
            continue
        for _ in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
            )
