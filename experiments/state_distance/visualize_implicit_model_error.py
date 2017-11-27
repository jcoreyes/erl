"""
Visualize how the errors in an implicitly learned dynamics model propagate over
time.

Usage:
```
python ../visualize_implicit_model_error.py path/to/params.pkl
```
"""
import argparse

import joblib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from railrl.policies.simple import UniformRandomPolicy
from railrl.state_distance.networks import NumpyModelExtractor


def visualize_policy_error(qf, env, horizon):
    model = NumpyModelExtractor(qf)
    policy = UniformRandomPolicy(env.action_space)
    actual_state = env.reset()

    predicted_states = []
    actual_states = []

    predicted_state = actual_state
    for _ in range(horizon):
        predicted_states.append(predicted_state.copy())
        actual_states.append(actual_state.copy())

        action, _ = policy.get_action(actual_state)
        predicted_state = model.next_state(predicted_state, action)
        actual_state = env.step(action)[0]

    predicted_states = np.array(predicted_states)
    actual_states = np.array(actual_states)
    times = np.arange(horizon)

    num_state_dims = env.observation_space.low.size
    dims = list(range(num_state_dims))
    norm = colors.Normalize(vmin=0, vmax=num_state_dims)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
    for dim in dims:
        plt.plot(
            times,
            predicted_states[:, dim],
            '--',
            label='Predicted, Dim {}'.format(dim),
            color=mapper.to_rgba(dim),
        )
        plt.plot(
            times,
            actual_states[:, dim],
            '-',
            label='Actual, Dim {}'.format(dim),
            color=mapper.to_rgba(dim),
        )
    plt.xlabel("Time Steps")
    plt.ylabel("Observation Value")
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=30, help='Horizon for eval')
    args = parser.parse_args()

    data = joblib.load(args.file)
    qf = data['qf']
    env = data['env']
    visualize_policy_error(qf, env, args.H)
