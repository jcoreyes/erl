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

from rlkit.policies.simple import RandomPolicy
import rlkit.torch.pytorch_util as ptu
from torch import optim


class NumpyModelExtractor(object):
    def __init__(
            self,
            qf,
            cheat,
            num_steps_left=0.,
            sample_size=100,
            learning_rate=1e-1,
            num_gradient_steps=100,
    ):
        super().__init__()
        self.qf = qf
        self.cheat = cheat
        self.num_steps_left = num_steps_left
        self.sample_size = sample_size
        self.learning_rate = learning_rate
        self.num_optimization_steps = num_gradient_steps

    def expand_to_sample_size(self, torch_array):
        return torch_array.repeat(self.sample_size, 1)

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.sample_size,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=False)

    def next_state(self, state, action):
        if self.cheat:
            next_states = self.qf.eval_np(
                observations=state[None],
                actions=action[None],
                goals=state[None],
                num_steps_left=np.array([[self.num_steps_left]]),
                return_predictions=True,
            )
            return next_states[0]
        num_steps_left = ptu.np_to_var(
            self.num_steps_left * np.ones((self.sample_size, 1))
        )
        obs_dim = state.shape[0]
        states = self.expand_np_to_var(state)
        actions = self.expand_np_to_var(action)
        next_states_np = np.zeros((self.sample_size, obs_dim))
        next_states = ptu.np_to_var(next_states_np, requires_grad=True)
        optimizer = optim.Adam([next_states], self.learning_rate)

        for _ in range(self.num_optimization_steps):
            losses = -self.qf(
                observations=states,
                actions=actions,
                goals=next_states,
                num_steps_left=num_steps_left,
            )
            loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses_np = ptu.get_numpy(losses)
        best_action_i = np.argmin(losses_np)
        return ptu.get_numpy(next_states[best_action_i, :])


def visualize_policy_error(qf, env, args):
    model = NumpyModelExtractor(qf, args.cheat, num_steps_left=args.tau)
    policy = RandomPolicy(env.action_space)
    actual_state = env.reset()

    predicted_states = []
    actual_states = []

    predicted_state = actual_state
    for _ in range(args.H):
        predicted_states.append(predicted_state.copy())
        actual_states.append(actual_state.copy())

        action, _ = policy.get_action(actual_state)
        predicted_state = model.next_state(predicted_state, action)
        actual_state = env.step(action)[0]

    predicted_states = np.array(predicted_states)
    actual_states = np.array(actual_states)
    times = np.arange(args.H)

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
    parser.add_argument('--tau', type=int, default=0)
    parser.add_argument('--cheat', action='store_true')
    args = parser.parse_args()

    if args.cheat and args.tau != 0:
        print("This setting doesn't make much sense. Are you sure?")

    data = joblib.load(args.file)
    qf = data['qf']
    env = data['env']
    visualize_policy_error(qf, env, args)
