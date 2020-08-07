import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import rlkit.torch.pytorch_util as ptu
from itertools import chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--plotv", action="store_true")
    args = parser.parse_args()

    path = args.path
    data = joblib.load(path)

    env = data['env']
    qf = data['qf']
    policy = data['policy']

    max_tau_nrow = 5
    max_tau_ncol = 5
    max_tau = max_tau_ncol * max_tau_nrow
    taus = list(range(0, max_tau))
    sample_size = 1000
    goal_state = env.sample_goal_for_rollout()
    goal_states = expand_np_to_var(goal_state, sample_size)

    """
    Try varying the state
    """
    if args.plotv:
        states = np.vstack([
            env.reset() for _ in range(sample_size)
        ])
        states = ptu.np_to_var(states)

        value_means = []
        all_values = []
        for tau in taus:
            expanded_tau = expand_np_to_var(tau, sample_size)
            actions = policy(states, goal_states, expanded_tau)
            values = ptu.get_numpy(qf(states, actions, goal_states, expanded_tau))
            all_values.append(values)
            value_means.append(np.mean(values))

        plt.figure()
        plt.plot(taus, value_means)
        plt.xlabel("Tau")
        plt.ylabel("V Value")
        plt.title("Mean V-Value vs Tau for sampled states")
        plt.show()

        fig, axes = plt.subplots(max_tau_nrow, max_tau_ncol)
        axes_flat = list(chain.from_iterable(axes))
        for tau, values, ax in zip(taus, all_values, axes_flat):
            ax.hist(values, bins=100)
            ax.set_title("Value histogram for sampled states, tau = {}".format(tau))
        plt.show()

    """
    Try varying the action
    """
    state = env.sample_states(1)[0]
    # state = env.sample_goal_state_for_rollout()
    states = expand_np_to_var(state, sample_size)

    q_value_maxs = []
    all_q_values = []
    actions = ptu.np_to_var(np.vstack([
        env.action_space.sample() for _ in range(sample_size)
    ]))
    for tau in taus:
        expanded_tau = expand_np_to_var(tau, sample_size)
        q_values = ptu.get_numpy(qf(states, actions, goal_states, expanded_tau))
        all_q_values.append(q_values)
        q_value_maxs.append(np.max(q_values))

    plt.figure()
    plt.plot(taus, q_value_maxs)
    plt.xlabel("Tau")
    plt.ylabel("Q value")
    plt.title("Max Q-Value vs Tau using sampled actions")
    plt.show()

    fig, axes = plt.subplots(max_tau_nrow, max_tau_ncol)
    axes_flat = list(chain.from_iterable(axes))
    for tau, q_values, ax in zip(taus, all_q_values, axes_flat):
        ax.hist(q_values, bins=100)
        ax.set_title(
            "Q-Value histogram for sampled actions, tau = {}".format(tau)
        )
    plt.show()


def expand_np_to_var(np_array, batch_size):
    return ptu.np_to_var(
        np.tile(
            np.expand_dims(np_array, 0),
            (batch_size, 1)
        )
    )


if __name__ == '__main__':
    main()
