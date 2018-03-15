import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from railrl.core import logger
from railrl.events.controllers import BetaLbfgsController

fig, (ax1, ax2) = plt.subplots(1, 2)


def debug(env, obs, agent_info):
    best_obs_seq = agent_info['best_obs_seq']
    best_action_seq = agent_info['best_action_seq']
    real_obs_seq = env.true_states(
        obs, best_action_seq
    )
    ax1.clear()
    env.plot_trajectory(
        ax1,
        np.array(best_obs_seq),
        np.array(best_action_seq),
        goal=env._target_position,
    )
    ax1.set_title("imagined")
    ax2.clear()
    env.plot_trajectory(
        ax2,
        np.array(real_obs_seq),
        np.array(best_action_seq),
        goal=env._target_position,
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
    reward_scale = variant['algo_kwargs'].get('reward_scale', 1)

    data = joblib.load(args.file)
    env = data['env']
    beta = data['beta_q']
    # beta_v = data['beta_v']
    lagrange_multiplier = args.lm / reward_scale
    planning_horizon = args.ph
    goal_slice = env.ob_to_goal_slice
    multitask_goal_slice = slice(None)
    optimizer = args.opt
    print("Optimizer choice: ", optimizer)
    print("lagrange multiplier: ", lagrange_multiplier)
    print("goal slice: ", goal_slice)
    print("multitask goal slice: ", multitask_goal_slice)
    policy = BetaLbfgsController(
        beta,
        beta,
        env,
        goal_slice=goal_slice,
        max_cost=128,
        learned_policy=data['policy'],
        multitask_goal_slice=multitask_goal_slice,
        planning_horizon=planning_horizon,
        replan_every_time_step=True,
        only_use_terminal_env_loss=False,
        use_learned_policy=False,
        solver_kwargs={
            'factr': 1e9,
        },
    )
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
