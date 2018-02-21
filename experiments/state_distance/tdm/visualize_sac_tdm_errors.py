import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


def main(args):

    data = joblib.load(args.file)
    variant_path = Path(args.file).parents[0] / 'variant.json'
    variant = json.load(variant_path.open())
    reward_scale = variant['sac_tdm_kwargs']['base_kwargs']['reward_scale']
    if args.pause:
        import ipdb; ipdb.set_trace()
    horizon = args.H
    env = data['env']
    qf = data['qf']
    vf = data['vf']
    policy = data['policy']
    goal = env.convert_ob_to_goal(env.reset())
    tau = np.array([[0]])
    state = env.reset()

    states = []

    q_vals = []
    v_vals = []
    a_vals = []
    distances = []
    log_probs = []

    for _ in range(horizon):
        states.append(state.copy())
        action, _ = policy.get_action(state, goal, tau[0])
        next_state, *_ = env.step(action)
        q_val = qf.eval_np(state[None], action[None], goal[None], tau)[0]
        v_val = vf.eval_np(state[None], goal[None], tau)[0]
        q_vals.append(q_val)
        v_vals.append(v_val)
        a_vals.append(q_val - v_val)
        distances.append(np.abs(
            env.convert_ob_to_goal(next_state) - goal
        ))
        sample_log_probs = []
        for _ in range(10):
            sample_log_probs.append(
                policy.eval_np(next_state[None], goal[None], tau, return_log_prob=True)[3]
            )
        log_probs.append(np.mean(sample_log_probs))

        state = next_state

    q_vals = np.array(q_vals).sum(1).flatten() / reward_scale
    v_vals = np.array(v_vals).sum(1).flatten() / reward_scale
    a_vals = np.array(a_vals).sum(1).flatten() / reward_scale
    distances = np.array(distances).sum(1).flatten()
    log_probs = np.array(log_probs).flatten()
    target_vf_outputs = - distances - log_probs
    plt.plot(q_vals, label='q values')
    plt.plot(v_vals, label='v values')
    plt.plot(a_vals, label='a values')
    plt.plot(-distances, label='negative distances')
    # plt.plot(log_probs, label='log_probs')
    # plt.plot(target_vf_outputs, label='target qf')
    plt.legend()
    plt.show()

PATH = '/home/vitchyr/git/railrl/data/doodads3/02-08-reacher7dof-sac-squared-distance-sweep-qf-activation-2/02-08-reacher7dof-sac-squared-distance-sweep-qf-activation-2-id1-s5793/params.pkl'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        type=str,
        # 2d point-mass
        # default='/home/vitchyr/git/railrl/data/local/02-01-dev-sac-tdm-launch/02-01-dev-sac-tdm-launch_2018_02_01_16_40_53_0000--s-2210/params.pkl',
        default=PATH,
        help='path to the snapshot file',
    )
    parser.add_argument('--H', type=int, default=30, help='Horizon for eval')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()

    main(args)
