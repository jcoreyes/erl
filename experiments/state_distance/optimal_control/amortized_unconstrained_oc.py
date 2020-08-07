"""
Amortized version of unconstrained_oc.py
"""
import argparse
import os
import random
from pathlib import Path

import joblib
import numpy as np
from rlkit.tf.state_distance.state_distance_q_learning import \
    multitask_rollout

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.multitask.reacher_7dof import (
    DESIRED_JOINT_CONFIG,
    reach_parameterized_joint_config)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.old.amortized_oc import \
    train_amortized_goal_chooser, AmortizedPolicy, UniversalGoalChooser
from rlkit.core import logger


def experiment(variant):
    num_rollouts = variant['num_rollouts']
    path = variant['qf_path']
    data = joblib.load(path)
    goal_conditioned_model = data['qf']
    env = data['env']
    argmax_qf_policy = data['policy']
    extra_data_path = Path(path).parent / 'extra_data.pkl'
    extra_data = joblib.load(str(extra_data_path))
    replay_buffer = extra_data['replay_buffer']


    """
    Train amortized policy
    """
    # goal_chooser = Mlp(
    #     output_size=env.goal_dim,
    #     input_size=int(env.observation_space.flat_dim),
    #     hidden_sizes=[100, 100],
    # )
    # goal_chooser = ReacherGoalChooser(
    #     hidden_sizes=[64, 64],
    # )
    goal_chooser = UniversalGoalChooser(
        input_goal_dim=7,
        output_goal_dim=env.goal_dim,
        obs_dim=int(env.observation_space.flat_dim),
        **variant['goal_chooser_params']
    )
    tau = variant['tau']
    if ptu.gpu_enabled():
        goal_chooser.to(ptu.device)
        goal_conditioned_model.to(ptu.device)
        argmax_qf_policy.to(ptu.device)
    train_amortized_goal_chooser(
        goal_chooser,
        goal_conditioned_model,
        argmax_qf_policy,
        tau,
        replay_buffer,
        **variant['train_params']
    )
    policy = AmortizedPolicy(argmax_qf_policy, goal_chooser)

    goal = np.array(variant['goal'])
    logger.save_itr_params(0, dict(
        env=env,
        policy=policy,
        goal_chooser=goal_chooser,
        goal=goal,
    ))
    """
    Eval policy.
    """
    paths = []
    # env.set_goal(goal)
    for _ in range(num_rollouts):
        # path = rollout(
        #     env,
        #     policy,
        #     **variant['rollout_params']
        # )
        # goal_expanded = np.expand_dims(goal, axis=0)
        # path['goal_states'] = goal_expanded.repeat(len(path['observations']), 0)
        goal = env.sample_goal_for_rollout()
        path = multitask_rollout(
            env,
            policy,
            goal,
            **variant['rollout_params']
        )
        paths.append(path)
    env.log_diagnostics(paths)
    logger.dump_tabular(with_timestamp=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file with a QF')
    parser.add_argument('--nrolls', type=int, default=10,
                        help='Number of rollouts to do.')
    parser.add_argument('--H', type=int, default=100, help='Horizon.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--discount', type=float, help='Discount Factor')
    parser.add_argument('--nsamples', type=int, default=1000,
                        help='Number of samples for optimization')
    parser.add_argument('--nups', type=int, default=10000,
                        help='Number of gradient updates')
    parser.add_argument('--dt', help='decrement tau', action='store_true')
    parser.add_argument('--cycle', help='cycle tau', action='store_true')
    parser.add_argument('--dc', help='decrement and cycle tau',
                        action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev"
    run_mode = 'none'
    use_gpu = True

    discount = 0
    if args.discount is not None:
        print("WARNING: you are overriding the discount factor. Right now "
              "only discount = 0 really makes sense.")
        discount = args.discount

    variant = dict(
        num_rollouts=args.nrolls,
        rollout_params=dict(
            max_path_length=args.H,
            animated=not args.hide,
            discount=discount,
            cycle_tau=args.cycle or args.dc,
            decrement_discount=args.dt or args.dc,
        ),
        policy_params=dict(
            sample_size=args.nsamples,
        ),
        qf_path=os.path.abspath(args.file),
        train_params=dict(
            learning_rate=1e-3,
            batch_size=32,
            num_updates=args.nups,
        ),
        # reward_function=reach_a_joint_config_reward_7dof,
        goal_chooser_params=dict(
            hidden_sizes=[100, 100],
            reward_function=reach_parameterized_joint_config,
        ),
        goal=list(DESIRED_JOINT_CONFIG),
        tau=5,
        # goal=list(REACH_A_POINT_GOAL),
        # reward_function=reach_a_point_reward,
        # reward_function=reach_a_joint_config_reward,
        # reward_function=hold_first_joint_and_move_second_joint_reward,
    )
    if run_mode == 'none':
        for exp_id in range(n_seeds):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                use_gpu=use_gpu,
            )
