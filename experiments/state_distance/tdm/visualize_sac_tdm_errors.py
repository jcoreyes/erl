import argparse
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path

from railrl.state_distance.util import merge_into_flat_obs


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
    policy.set_goal(goal)
    policy.set_tau(0)
    state = env.reset()

    states = []

    q_vals = []
    v_vals = []
    a_vals = []
    distances = []
    log_probs = []

    for _ in range(horizon):
        states.append(state.copy())
        action, _ = policy.get_action(state)
        next_state, *_ = env.step(action)
        flat_ob = merge_into_flat_obs(state, goal, np.array([0]))
        q_val = qf.eval_np(flat_ob[None], action[None])[0]
        v_val = vf.eval_np(flat_ob[None])[0]
        q_vals.append(q_val)
        v_vals.append(v_val)
        a_vals.append(q_val - v_val)
        distances.append(np.abs(
            env.convert_ob_to_goal(next_state)
            - env.convert_ob_to_goal(goal)
        ))
        sample_log_probs = []
        next_flat_ob = merge_into_flat_obs(next_state, goal, np.array([0]))
        for _ in range(10):
            sample_log_probs.append(
                policy.eval_np(next_flat_ob[None], return_log_prob=True)[3]
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
    plt.plot(log_probs, label='log_probs')
    plt.plot(target_vf_outputs, label='target qf')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        type=str,
        # 2d point-mass
        default='/home/vitchyr/git/railrl/data/local/02-01-dev-sac-tdm-launch/02-01-dev-sac-tdm-launch_2018_02_01_16_40_53_0000--s-2210/params.pkl',
        help='path to the snapshot file',
    )
    parser.add_argument('--H', type=int, default=30, help='Horizon for eval')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()

    main(args)
