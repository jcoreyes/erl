import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

import joblib
import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.core import logger
from railrl.torch.core import PyTorchModule
from railrl.torch.mpc.collocation.collocation_mpc_controller import (
    SlsqpCMC,
    GradientCMC,
    StateGCMC,
    LBfgsBCMC,
    BfgsBCMC,
    LBfgsBStateOnlyCMC)


class TdmToImplicitModel(PyTorchModule):
    def __init__(self, env, qf, tau):
        self.quick_init(locals())
        super().__init__()
        self.env = env
        self.qf = qf
        self.tau = tau

    def forward(self, states, actions, next_states):
        taus = ptu.np_to_var(
            self.tau * np.ones((states.shape[0], 1))
        )
        goals = self.env.convert_obs_to_goals(next_states)
        return self.qf(
            observations=states,
            actions=actions,
            goals=goals,
            num_steps_left=taus,
        ).sum(1)


class TdmPolicyToTimeInvariantGoalReachingPolicy(PyTorchModule):
    def __init__(self, tdm_policy, env, num_steps_left):
        self.quick_init(locals())
        super().__init__()
        self.tdm_policy = tdm_policy
        self.env = env
        self.num_steps_left = num_steps_left

    def forward(self, states, next_states):
        num_steps_left = ptu.np_to_var(
            self.num_steps_left * np.ones((states.shape[0], 1))
        )
        goals = self.env.convert_obs_to_goals(next_states)
        return self.tdm_policy(
            observations=states,
            goals=goals,
            num_steps_left=num_steps_left,
        )[0]


class TrueModelToImplicitModel(PyTorchModule):
    def __init__(self, env):
        self.quick_init(locals())
        super().__init__()
        self.env = env

    def forward(self, states, actions, next_states):
        state = ptu.get_numpy(states[0])
        action = ptu.get_numpy(actions[0])
        next_state = ptu.get_numpy(next_states[0])

        true_next_state = self.env.true_model(state, action)
        return -((next_state - true_next_state)**2).sum()


fig, (ax1, ax2) = plt.subplots(1, 2)


def debug(env, obs, agent_info):
    best_obs_seq = agent_info['best_obs_seq']
    best_action_seq = agent_info['best_action_seq']
    real_obs_seq = env.wrapped_env.true_states(
        obs, best_action_seq
    )
    ax1.clear()
    env.wrapped_env.plot_trajectory(
        ax1,
        np.array(best_obs_seq),
        np.array(best_action_seq),
        goal=env.wrapped_env._target_position,
    )
    ax1.set_title("imagined")
    ax2.clear()
    env.wrapped_env.plot_trajectory(
        ax2,
        np.array(real_obs_seq),
        np.array(best_action_seq),
        goal=env.wrapped_env._target_position,
    )
    ax2.set_title("real")
    plt.draw()
    plt.pause(0.001)


def rollout(env, agent, max_path_length=np.inf, animated=False, tick=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        debug(env, o, agent_info)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
        if tick:
            import ipdb; ipdb.set_trace()

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--ph', type=int, default=3,
                        help='planning horizon')
    parser.add_argument('--nrolls', type=int, default=1,
                        help='Number of rollout per eval')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--lm', type=float, default=1,
                        help='Lagrange Multiplier (before division by reward scale)')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--tick', action='store_true')
    parser.add_argument('--justsim', action='store_true')
    parser.add_argument('--npath', type=int, default=100)
    parser.add_argument('--tau', type=int, default=0)
    parser.add_argument('--opt', type=str, default='lbfgs')
    args = parser.parse_args()
    if args.pause:
        import ipdb; ipdb.set_trace()

    variant_path = Path(args.file).parents[0] / 'variant.json'
    variant = json.load(variant_path.open())
    reward_scale = variant['sac_tdm_kwargs']['base_kwargs']['reward_scale']

    data = joblib.load(args.file)
    env = data['env']
    qf = data['qf']

    implicit_model = TdmToImplicitModel(
        env,
        qf,
        tau=args.tau,
    )
    # implicit_model = TrueModelToImplicitModel(env)
    lagrange_multiplier = args.lm / reward_scale
    # lagrange_multiplier = 10
    planning_horizon = args.ph
    goal_slice = env.ob_to_goal_slice
    multitask_goal_slice = slice(None)
    optimizer = args.opt
    print("Optimizer choice: ", optimizer)
    print("lagrange multiplier: ", lagrange_multiplier)
    print("goal slice: ", goal_slice)
    print("multitask goal slice: ", multitask_goal_slice)
    if optimizer == 'slsqp':
        policy = SlsqpCMC(
            implicit_model,
            env,
            goal_slice=goal_slice,
            multitask_goal_slice=multitask_goal_slice,
            planning_horizon=planning_horizon,
            # use_implicit_model_gradient=True,
            solver_params={
                'ftol': 1e-2,
                'maxiter': 100,
            },
        )
    elif optimizer == 'gradient':
        policy = GradientCMC(
            implicit_model,
            env,
            goal_slice=goal_slice,
            multitask_goal_slice=multitask_goal_slice,
            planning_horizon=planning_horizon,
            lagrange_multiplier=lagrange_multiplier,
            num_grad_steps=100,
            num_particles=128,
            warm_start=False,
        )
    elif optimizer == 'state':
        policy = StateGCMC(
            implicit_model,
            env,
            goal_slice=goal_slice,
            multitask_goal_slice=multitask_goal_slice,
            planning_horizon=planning_horizon,
            lagrange_multiplier=lagrange_multiplier,
            num_grad_steps=100,
            num_particles=128,
            warm_start=False,
        )
    elif optimizer == 'lbfgs':
        policy = LBfgsBCMC(
            implicit_model,
            env,
            goal_slice=goal_slice,
            multitask_goal_slice=multitask_goal_slice,
            lagrange_multipler=lagrange_multiplier,
            planning_horizon=planning_horizon,
            # finite_difference=True,
            solver_params={
                'factr': 1e9,
            },
        )
    elif optimizer == 'slbfgs':
        universal_policy = TdmPolicyToTimeInvariantGoalReachingPolicy(
            tdm_policy=data['policy'],
            env=env,
            num_steps_left=args.tau,
        )
        policy = LBfgsBStateOnlyCMC(
            implicit_model,
            env,
            universal_policy,
            goal_slice=goal_slice,
            multitask_goal_slice=multitask_goal_slice,
            lagrange_multipler=lagrange_multiplier,
            planning_horizon=planning_horizon,
            # finite_difference=True,
            solver_params={
                'factr': 1e9,
            },
        )
    elif optimizer == 'bfgs':
        policy = BfgsBCMC(
            implicit_model,
            env,
            goal_slice=goal_slice,
            multitask_goal_slice=multitask_goal_slice,
            lagrange_multipler=lagrange_multiplier,
            planning_horizon=planning_horizon,
            # solver_params={},
        )
    else:
        raise ValueError("Unknown optimizer type: {}".format(optimizer))
    paths = []
    while True:
        env.set_goal(env.sample_goal_for_rollout())
        paths.append(rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=not args.hide,
            tick=args.tick,
        ))
        env.log_diagnostics(paths)
        logger.dump_tabular()
