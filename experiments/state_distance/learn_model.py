import argparse
import random

from rlkit.tf.state_distance.model_learning import ModelLearning
from rlkit.tf.state_distance.util import get_replay_buffer
from torch.nn import functional as F

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.multitask.reacher_env import (
    GoalStateSimpleStateReacherEnv,
)
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.old.model_based_policies import MultistepModelBasedPolicy
from rlkit.tf.predictors import Mlp


def experiment(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    env = NormalizedBoxEnv(
        env,
        **variant['normalize_params']
    )
    if variant['start_with_empty_replay_buffer']:
        replay_buffer = None
    else:
        replay_buffer = get_replay_buffer(variant)
    model_learns_deltas = variant['model_learns_deltas']

    observation_space = env.observation_space
    action_space = env.action_space
    model = Mlp(
        int(observation_space.flat_dim) + int(action_space.flat_dim),
        int(observation_space.flat_dim),
        **variant['model_params']
    )
    policy = MultistepModelBasedPolicy(
        model,
        env,
        model_learns_deltas=model_learns_deltas,
        **variant['policy_params']
    )
    algo = ModelLearning(
        env,
        model,
        replay_buffer=replay_buffer,
        eval_policy=policy,
        model_learns_deltas=model_learns_deltas,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.to(ptu.device)
    algo.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_path', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-reacher-model-learning"
    version = "Dev"
    run_mode = "none"

    # n_seeds = 3
    # mode = "ec2"
    exp_prefix = "local-reacher-2d-learn-small-model-softplus-match-magic-params"
    # run_mode = 'custom_grid'

    num_configurations = 1  # for random mode
    snapshot_mode = "last"
    snapshot_gap = 5
    use_gpu = True
    if mode != "local":
        use_gpu = False

    dataset_path = args.replay_path

    # noinspection PyTypeChecker
    max_path_length = 300
    replay_buffer_size = 100000
    variant = dict(
        dataset_path=str(dataset_path),
        algo_params=dict(
            num_epochs=100,
            num_batches_per_epoch=1000,
            batch_size=100,
            learning_rate=1e-3,
            weight_decay=0.0001,
            max_path_length=max_path_length,
            replay_buffer_size=replay_buffer_size,
            add_on_policy_data=True,
        ),
        policy_params=dict(
            sample_size=10000,
            planning_horizon=6,
            action_penalty=0.0001,
        ),
        model_learns_deltas=True,
        model_params=dict(
            hidden_activation=F.softplus,
            hidden_sizes=[100, 100],
        ),
        env_class=GoalStateSimpleStateReacherEnv,
        # env_class=Reacher7DofFullGoalState,
        # env_class=PusherEnv,
        # env_class=XyMultitaskSimpleStateReacherEnv,
        # env_class=MultitaskPoint2DEnv,
        env_params=dict(
            # add_noop_action=False,
        ),
        normalize_params=dict(
            obs_mean=None,
            obs_std=[0.7, 0.3, 0.7, 0.3, 25, 5],
            # obs_std=[3, 3],
        ),
        sampler_params=dict(
            min_num_steps_to_collect=10000,
            max_path_length=max_path_length,
            render=args.render,
        ),
        replay_buffer_size=replay_buffer_size,
        sampler_es_class=OUStrategy,
        # sampler_es_class=UniformStrategy,
        sampler_es_params=dict(
            max_sigma=0.2,
            min_sigma=0.2,
        ),
        generate_data=args.replay_path is None,
        start_with_empty_replay_buffer=False,
    )
    if run_mode == 'grid':
        search_space = {
            'model_params.hidden_sizes':[
                [100, 100],
                [500, 500],
            ],
            'algo_params.num_batches_per_epoch': [1000, 10000],
            # 'algo_params.weight_decay': [0, 1e-4, 1e-3, 1e-2],
            # 'normalize_params.obs_std': [
                # None,
                # [0.7, 0.3, 0.7, 0.3, 25, 5],
            # ],
            # 'model_learns_deltas': [True, False],
        }
        sweeper = hyp.DeterministicHyperparameterSweeper(
            search_space, default_parameters=variant,
        )
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    use_gpu=use_gpu,
                    snapshot_mode=snapshot_mode,
                    snapshot_gap=snapshot_gap,
                )
    elif run_mode == 'custom_grid':
        for exp_id, (
                add_on_policy_data,
                on_policy_num_steps,
                off_policy_num_steps,
        ) in enumerate([
            (False, 0, 10000),
            (False, 0, 100000),
            # (True, 50000, 50000),
            (True, 90000, 10000),
            # (True, 10000, 10000),
            # (True, 10000, 0),
        ]):
            variant['algo_params']['add_on_policy_data'] = add_on_policy_data
            variant['algo_params']['max_num_on_policy_steps_to_add'] = (
                on_policy_num_steps
            )
            variant['sampler_params']['min_num_steps_to_collect'] = (
                off_policy_num_steps
            )
            variant['version'] = "off{}k_on{}k".format(
                off_policy_num_steps // 1000,
                on_policy_num_steps // 1000,
            )
            for _ in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    use_gpu=use_gpu,
                    snapshot_mode=snapshot_mode,
                    snapshot_gap=snapshot_gap,
                )
    else:
        for _ in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=0,
                use_gpu=use_gpu,
                snapshot_mode=snapshot_mode,
                snapshot_gap=snapshot_gap,
            )
