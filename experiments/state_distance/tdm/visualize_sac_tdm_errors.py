import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt

from railrl.state_distance.util import merge_into_flat_obs


def main(args):
    data = joblib.load(args.file)
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

    qf_outputs = []
    distances = []
    entropies = []

    for _ in range(horizon):
        states.append(state.copy())
        action, _ = policy.get_action(state)
        next_state, *_ = env.step(action)
        flat_ob = merge_into_flat_obs(state, goal, np.array([0]))
        qf_outputs.append(
            qf.eval_np(flat_ob[None], action[None])[0]
            - vf.eval_np(flat_ob[None])[0]
        )
        distances.append(
            env.convert_ob_to_goal(state)
            - env.convert_ob_to_goal(next_state)
        )
        sample_log_probs = []
        for _ in range(10):
            sample_log_probs.append(
                policy.eval_np(flat_ob[None], return_log_prob=True)[3]
            )
        entropies.append(-np.mean(np.vstack(sample_log_probs), axis=0))

    qf_outputs = np.array(qf_outputs).sum(1).flatten()
    distances = np.array(distances).sum(1).flatten()
    entropies = np.array(entropies).flatten()
    target_qf_outputs = distances + entropies
    plt.plot(qf_outputs, label='qf')
    plt.plot(distances, label='distances')
    plt.plot(entropies, label='entropies')
    plt.plot(target_qf_outputs, label='target qf')
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
