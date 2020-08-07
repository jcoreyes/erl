import argparse
import matplotlib.pyplot as plt

import joblib
import numpy as np
import torch
from torch import optim

from rlkit.state_distance.util import merge_into_flat_obs
from rlkit.torch import pytorch_util as ptu
from rlkit.misc.eval_util import get_generic_path_information
from rlkit.core import logger


def find_reachable_goal_state(env, ob, tau, qf, policy):
    _obs = ob[None]
    _taus = np.array([[tau]])
    _goals = env.convert_ob_to_goal(ob)[None]
    _goals = 10 * np.ones_like(_goals)
    obs = ptu.np_to_var(_obs, requires_grad=False)
    goals = ptu.np_to_var(_goals, requires_grad=True)
    taus = ptu.np_to_var(_taus, requires_grad=False)
    optimizer = optim.Adam([goals], 1e-1)
    for _ in range(100):
        flat_obs = torch.cat((obs, goals, taus), dim=1)
        actions = policy(flat_obs)
        loss = - qf(flat_obs, actions).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return ptu.get_numpy(goals)


def rollout(env, agent, qf, init_tau, max_path_length=np.inf,
                          animated=False, cycle_tau=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    taus = []
    o = env.reset()
    next_o = None
    path_length = 0
    tau = init_tau
    if animated:
        env.render()
    current_xy = env.convert_ob_to_goal(o)
    delta = env.multitask_goal - current_xy
    reachable_to_current_distances = []
    tdm_distances = []
    while path_length < max_path_length:
        current_xy = env.convert_ob_to_goal(o)
        # new_goal = current_xy + delta
        new_goal = env.multitask_goal
        env.set_goal(new_goal)
        agent.set_goal(new_goal)
        a, agent_info = agent.get_action(o)
        flat_obs = merge_into_flat_obs(o, new_goal, np.array([tau]))
        pred_xy = qf.eval_np(flat_obs[None], a[None], return_internal_prediction=True)
        tdm_values = qf.eval_np(flat_obs[None], a[None])
        reachable_goal = find_reachable_goal_state(env, o, tau, qf, agent)
        print("tau", tau)
        # print("new_goal", new_goal)
        # print("pred_xy", pred_xy)
        # print("tdm values", tdm_values)
        # print("tdm distance", np.sum(np.abs(tdm_values)))
        # print("actual distance", sum(np.abs(current_xy - new_goal)))
        print("current_xy", current_xy)
        print("reachable xy", reachable_goal)
        reachable_to_current_distances.append(
            np.linalg.norm(current_xy - reachable_goal)
        )
        tdm_distances.append(
            np.sum(np.abs(tdm_values))
        )
        next_o, r, d, env_info = env.step(a)
        agent.set_tau(tau)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        taus.append(np.array([tau]))
        path_length += 1
        if cycle_tau:
            tau -= 1
            if tau < 0:
                tau = init_tau
        if d:
            break
        o = next_o
        if animated:
            env.render()
            # input("Press Enter to continue...")

    plt.subplot(2, 1, 1)
    plt.plot(np.array(reachable_to_current_distances))
    plt.subplot(2, 1, 2)
    plt.plot(np.array(tdm_distances))
    plt.show()

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
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
        # final_observation=next_o,
        num_steps_left=np.array(taus),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--nrolls', type=int, default=1,
                        help='Number of rollout per eval')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--mtau', type=float,
                        help='Max tau value')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--cycle', help='cycle tau', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    if 'policy' in data:
        policy = data['policy']
    else:
        policy = data['exploration_policy']
    qf = data['qf']
    policy.train(False)
    qf.train(False)

    if args.pause:
        import ipdb; ipdb.set_trace()

    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)

    if args.mtau is None:
        print("Defaulting max tau to 10.")
        max_tau = 10
    else:
        max_tau = args.mtau

    while True:
        paths = []
        for _ in range(args.nrolls):
            goal = env.sample_goal_for_rollout()
            print("goal", goal)
            env.set_goal(goal)
            policy.set_goal(goal)
            policy.set_tau(max_tau)
            path = rollout(
                env,
                policy,
                qf,
                init_tau=max_tau,
                max_path_length=args.H,
                animated=not args.hide,
                cycle_tau=args.cycle,
            )
            paths.append(path)
        env.log_diagnostics(paths)
        for key, value in get_generic_path_information(paths).items():
            logger.record_tabular(key, value)
        logger.dump_tabular()

if __name__ == '__main__':
    main()
