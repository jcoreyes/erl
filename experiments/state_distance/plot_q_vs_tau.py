import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import railrl.torch.pytorch_util as ptu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    path = args.path
    data = joblib.load(path)

    env = data['env']
    qf = data['qf']
    policy = data['policy']

    states = env.sample_goal_states(100)
    goal_state = env.sample_goal_state_for_rollout()
    value_means = []
    taus = list(range(0, 100))
    batch_size = states.shape[0]
    states = ptu.np_to_var(states)
    goal_states = expand_np_to_var(goal_state, batch_size)
    for tau in taus:
        expanded_tau = expand_np_to_var(tau, batch_size)
        actions = policy(states, goal_states, expanded_tau)
        value = qf(states, actions, goal_states, expanded_tau)
        value_means.append(np.mean(ptu.get_numpy(value)))
    plt.plot(taus, value_means)
    plt.show()
    plt.xlabel("Tau")
    plt.ylabel("Q-value")


def expand_np_to_var(np_array, batch_size):
    return ptu.np_to_var(
        np.tile(
            np.expand_dims(np_array, 0),
            (batch_size, 1)
        )
    )


if __name__ == '__main__':
    main()
