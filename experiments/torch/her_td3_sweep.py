import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYZEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import (
    SawyerReachXYEnv, SawyerReachXYZEnv)
from railrl.data_management.her_replay_buffer import RelabelingReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from railrl.launchers.experiments.vitchyr.multiworld import her_td3_experiment
from railrl.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=100,
            batch_size=128,
            discount=0.99,
            replay_buffer_size=int(1E6),
            num_updates_per_env_step=4,
            desired_goal_key='desired_goal'
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        exploration_type='ou',
        es_kwargs=dict(
            max_sigma=0.1,
        ),
        replay_buffer_class=ObsDictRelabelingBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
            # fraction_goals_are_rollout_goals=1,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        algorithm="HER-TD3",
        version="normal",
        env_kwargs=dict(
            fix_goal=False,
            # fix_goal=True,
            # fixed_goal=(0, 0.7),
        ),
        normalize=False,
        observation_key='observation',
        desired_goal_key='desired_goal',
    )
    search_space = {
        'env_class': [
            # SawyerPushAndReachXYZEnv,
            # SawyerPushAndReachXYEnv,
            # SawyerReachXYZEnv,
            SawyerReachXYEnv,
        ],
        'env_kwargs.reward_type': [
            # 'hand_and_puck_distance',
            # 'hand_and_puck_success',
            'puck_distance',
            # 'puck_success',
            # 'hand_distance',
            # 'hand_success',
        ],
        'algo_kwargs.discount': [0.98, 0.99],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    # mode = 'here_no_doodad'
    exp_prefix = 'dev'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'multiworld-goal-env-her-td3-new-gripper-sweep-discount'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
            )
