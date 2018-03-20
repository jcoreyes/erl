import argparse
import joblib

import railrl.torch.pytorch_util as ptu
import railrl.misc.hyperparameter as hyp
from railrl.data_management.her_replay_buffer import HerReplayBuffer, \
    PrioritizedHerReplayBuffer, SimplePrioritizedHerReplayBuffer
from railrl.envs.multitask.point2d import MultitaskPoint2DEnv
from railrl.envs.multitask.point2d_wall import MultitaskPoint2dWall
from railrl.events.beta_learning import BetaLearning
from railrl.events.controllers import BetaLbfgsController, BetaMultigoalLbfgs
from railrl.events.networks import BetaQ, TanhFlattenMlpPolicy, BetaV
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_and_epislon import \
    GaussianAndEpislonStrategy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.launchers.launcher_util import setup_logger, run_experiment
from railrl.policies.simple import RandomPolicy
from railrl.torch.networks import TanhMlpPolicy


def experiment(variant):
    env = variant['env_class']()
    es = GaussianAndEpislonStrategy(
    # es = GaussianStrategy(
        action_space=env.action_space,
        **variant['es_kwargs']
    )
    beta_q = BetaQ(
        env,
        False,
        hidden_sizes=[32, 32],
    )
    beta_q2 = BetaQ(
        env,
        False,
        hidden_sizes=[32, 32],
    )
    beta_v = BetaV(
        env,
        False,
        hidden_sizes=[32, 32],
    )
    policy = TanhFlattenMlpPolicy(
        env,
        hidden_sizes=[32, 32],
    )
    if variant['load_file'] is not None:
        data = joblib.load(variant['load_file'])
        beta_q = data['beta_q']
        beta_q2 = data['beta_q2']
        beta_v = data['beta_v']
        policy = data['policy']

    goal_slice = env.ob_to_goal_slice
    multitask_goal_slice = slice(None)
    controller = BetaMultigoalLbfgs(
        beta_q,
        beta_v,
        env,
        goal_slice=goal_slice,
        learned_policy=policy,
        multitask_goal_slice=multitask_goal_slice,
        **variant['controller_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        # policy=policy,
        # policy=RandomPolicy(env.action_space),
        policy=controller,
    )
    replay_buffer = SimplePrioritizedHerReplayBuffer(
        env=env,
        **variant['replay_buffer_kwargs']
    )
    algorithm = BetaLearning(
        env,
        exploration_policy=exploration_policy,
        beta_q=beta_q,
        beta_q2=beta_q2,
        beta_v=beta_v,
        policy=policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--mt', type=int, help='max time to goal', default=0)
    args = parser.parse_args()

    n_seeds = 1
    exp_prefix = "dev-beta-learning"

    # n_seeds = 3
    # exp_prefix = "beta-learning-uwall-smart-her-wall-random-exploration" \
    #              "-random-goals-mtau-2"

    variant = dict(
        # env_class=MultitaskPoint2DEnv,
        load_file=args.file,
        env_class=MultitaskPoint2dWall,
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=500,
            num_steps_per_eval=50,
            num_updates_per_env_step=1,
            max_path_length=25,
            batch_size=512,
            discount=0.,
            prioritized_replay=False,
            render=True,
            render_during_eval=True,
            # save_replay_buffer=True,

            finite_horizon=True,
            max_num_steps_left=1,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E5),
            num_goals_to_sample=2,
            max_time_to_next_goal=2,
            fraction_goals_are_rollout_goals=1,
            # resampling_strategy='truncated_geometric',
            # truncated_geom_factor=0.5,
        ),
        controller_kwargs=dict(
            max_cost=128,
            planning_horizon=5,
            replan_every_time_step=True,
            only_use_terminal_env_loss=False,
            use_learned_policy=True,
            solver_kwargs={
                'factr': 1e5,
            },
        ),
        es_kwargs=dict(
            epsilon=0.2,
            max_sigma=0.,
        )
    )
    search_space = {
        # 'algo_kwargs.prioritized_replay': [True, False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for i in range(n_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                experiment,
                mode='local',
                exp_prefix=exp_prefix,
                variant=variant,
                exp_id=exp_id,
                # snapshot_gap=1,
                # snapshot_mode='gap',
            )
