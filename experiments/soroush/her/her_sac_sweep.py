import argparse

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.experiments.soroush.multiworld_her import her_sac_experiment
import rlkit.misc.hyperparameter as hyp

variant = dict(
    algo_kwargs=dict(
        num_epochs=300,
        num_steps_per_epoch=1000,
        num_steps_per_eval=1000,
        max_path_length=100,
        batch_size=128,
        discount=0.99,
        num_updates_per_env_step=1,
        soft_target_tau=1e-3,  # 1e-2
        target_update_period=1,  # 1
        train_policy_with_reparameterization=True,
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
    replay_buffer_kwargs=dict(
        max_size=int(1E6),
        fraction_goals_are_rollout_goals=0.2,
        fraction_resampled_goals_are_env_goals=0.5,
    ),
    algorithm="HER-SAC",
    version="normal",
    env_kwargs=dict(
        fix_goal=False,
        # fix_goal=True,
        # fixed_goal=(0, 0.7),
    ),
    normalize=False,
)

common_params = {
    # 'normalize': [False, True],
}

env_params = {
    'sawyer-reach-xy': { # 6 DoF
        'env_class': [SawyerReachXYEnv],
        'env_kwargs.reward_type': ['hand_distance'],
        'algo_kwargs.num_epochs': [50],
        'algo_kwargs.reward_scale': [1e0, 1e1, 1e2, 1e3] #[0.01, 0.1, 1, 10, 100],
    },
    'sawyer-push-and-reach-xy': {  # 6 DoF
        'env_class': [SawyerPushAndReachXYEnv],
        'env_kwargs.reward_type': ['puck_distance'],
        'algo_kwargs.discount': [0.98, 0.99],
        'algo_kwargs.num_updates_per_env_step': [4],
        'algo_kwargs.num_epochs': [1000],
        'algo_kwargs.reward_scale': [1e0, 1e1, 1e2, 1e3],  # [0.01, 0.1, 1, 10, 100],
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default='sawyer-reach-xy')
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = parse_args()

    exp_prefix = "her-sac-" + args.env
    if len(args.label) > 0:
        exp_prefix = exp_prefix + "-" + args.label

    search_space = common_params
    search_space.update(env_params[args.env])
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for _ in range(args.num_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                her_sac_experiment,
                exp_prefix=exp_prefix,
                mode=args.mode,
                exp_id=exp_id,
                variant=variant,
                use_gpu=args.gpu,
                snapshot_gap=int(variant['algo_kwargs']['num_epochs'] / 10),
                snapshot_mode='gap_and_last',
            )
