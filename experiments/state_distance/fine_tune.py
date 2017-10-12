"""
Fine tune a trained policy/qf
"""
import argparse
import random

import joblib

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    HorizonFedStateDistanceQLearning)
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import ConstantSchedule, StatConditionalSchedule
from railrl.torch.modules import HuberLoss
from railrl.torch.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy


def experiment(variant):
    path = variant['path']
    data = joblib.load(path)
    env = data['env']
    qf = data['qf']
    policy = data['policy']
    action_space = convert_gym_space(env.action_space)
    es = variant['es_class'](
        action_space=action_space,
        **variant['es_params']
    )
    exploration_policy = UniversalPolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    qf_criterion = variant['qf_criterion_class'](
        **variant['qf_criterion_params']
    )
    epoch_discount_schedule_class = variant['epoch_discount_schedule_class']
    epoch_discount_schedule = epoch_discount_schedule_class(
        **variant['epoch_discount_schedule_params']
    )
    algo = variant['algo_class'](
        env,
        qf,
        policy,
        exploration_policy,
        epoch_discount_schedule=epoch_discount_schedule,
        qf_criterion=qf_criterion,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
                        help='Path to snapshot file to fine tune.')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-sdql-fine-tune"
    run_mode = "none"

    n_seeds = 1
    # mode = "ec2"
    exp_prefix = "sdql-fine-tune"
    # run_mode = 'grid'

    num_configurations = 50  # for random mode
    snapshot_mode = "gap"
    snapshot_gap = 25
    use_gpu = True
    if mode != "local":
        use_gpu = False

    max_path_length = 300
    # noinspection PyTypeChecker
    variant = dict(
        path=args.path,
        algo_params=dict(
            num_epochs=101,
            num_steps_per_epoch=300,
            num_steps_per_eval=3000,
            num_updates_per_env_step=50,
            use_soft_update=True,
            tau=0.001,
            batch_size=500,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            sample_goals_from='environment',
            # sample_goals_from='replay_buffer',
            sample_discount=True,
            qf_weight_decay=0.,
            max_path_length=max_path_length,
            use_new_data=True,
            replay_buffer_size=1000000,
            prob_goal_state_is_next_state=0,
            termination_threshold=0,
            sparse_reward=True,
            save_replay_buffer=True,
        ),
        algo_class=HorizonFedStateDistanceQLearning,
        # epoch_discount_schedule_class=IntRampUpSchedule,
        epoch_discount_schedule_class=StatConditionalSchedule,
        epoch_discount_schedule_params=dict(
            init_value=5,
            stat_bounds=(0.06, 0.09),
            running_average_length=1,
            delta=-1,
            value_bounds=(5, None),
            statistic_name="Final Euclidean distance to goal Mean",
            # min_value=0,
            # max_value=100,
            # ramp_duration=50,
        ),
        qf_criterion_class=HuberLoss,
        qf_criterion_params=dict(),
        es_class=OUStrategy,
        es_params=dict(
            theta=0.15,
            max_sigma=0.2,
            min_sigma=0.2,
        ),
    )
    if run_mode == 'grid':
        search_space = {
            'epoch_discount_schedule_params.value': [10, 50, 0],
            'algo_params.optimize_target_policy': [True, False],
            'algo_params.residual_gradient_weight': [0.5, 0],
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
