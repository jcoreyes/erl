from rlkit.launchers.launcher_util import run_experiment
import rlkit.misc.hyperparameter as hyp
import argparse
import math

from rlkit.launchers.exp_launcher import rl_experiment

from multiworld.envs.mujoco.cameras import *

from rlkit.launchers.arglauncher import run_variants

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_leap import SawyerPushAndReachXYEnv

variant = dict(
    rl_variant=dict(
        do_state_exp=True,
        algo_kwargs=dict(
            num_epochs=1000,
            batch_size=2048, #128,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000, #4000,
            min_num_steps_before_training=1000,
        ),
        max_path_length=100,
        td3_trainer_kwargs=dict(
            discount=0.995,
        ),
        twin_sac_trainer_kwargs=dict(
            discount=0.995,
        ),
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
        use_subgoal_policy=False,
        subgoal_policy_kwargs=dict(
            num_subgoals_per_episode=2,
        ),
        use_masks=False,
        exploration_type='gaussian_and_epsilon',
        es_kwargs=dict(
            max_sigma=.2,
            min_sigma=None,
            epsilon=.3,
        ),
        algorithm="TD3",
        dump_video_kwargs=dict(
            rows=1,
            columns=8,
        ),
        vis_kwargs=dict(
            vis_list=dict(),
        ),
        save_video_period=50,

        expl_mask_distribution_kwargs=dict(
            items=(
                (1, 1, 0, 0), # hand
                (0, 0, 1, 1), # puck
                (1, 1, 1, 1), # hand and puck
            ),
            weights=(1,1,1)
        ),

        eval_mask_distribution_kwargs=dict(
            items=(
                (1, 1, 0, 0), # hand
                (0, 0, 1, 1), # puck
                (1, 1, 1, 1), # hand and puck
            ),
            weights=(1,1,1)
        ),

        observation_key="state_observation",
        desired_goal_key="mask_desired_goal",
        achieved_goal_key="mask_achieved_goal",
    ),
    env_class=SawyerPushAndReachXYEnv,
    env_kwargs=dict(
        hand_low=(-0.20, 0.50),
        hand_high=(0.20, 0.70),
        puck_low=(-0.20, 0.50),
        puck_high=(0.20, 0.70),
        goal_low=(-0.20, 0.50, -0.20, 0.50),
        goal_high=(0.20, 0.70, 0.20, 0.70),
        fix_reset=False,
        sample_realistic_goals=False,
        reward_type='hand_and_puck_distance',
        invisible_boundary_wall=True,
    ),

    num_exps_per_instance=1,
    region='us-west-2',
)

if __name__ == "__main__":
    search_space = {
        'env_kwargs.reward_type': ['hand_and_puck_distance', ],
        'rl_variant.use_masks': [True, ],
        'rl_variant.max_path_length': [200],
        'init_camera': [sawyer_xyz_reacher_camera_v0],
        'rl_variant.vis_kwargs.vis_list': [[],],
            # 'plt',
        # ]],
        'rl_variant.algo_kwargs.num_trains_per_train_loop': [4000],
        'rl_variant.es_kwargs.max_sigma': [0.8],
        'rl_variant.td3_trainer_kwargs.discount': [0.99, ],

        'rl_variant.expl_mask_distribution_kwargs.weights': [
            # (1,1,1),
            # (0,1,1),
            # (0,0,1),
            (0,1,0),
        ],
        'rl_variant.eval_mask_distribution_kwargs.weights': [
            # (0,0,1),
            (0,1,0),
        ],
        'seedid': range(5),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(rl_experiment, variants, run_id=0)
