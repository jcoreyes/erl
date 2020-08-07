"""
Visualize how good a model is at predicted the future time steps.

Usage:
```
python ../this_script.py path/to/params.pkl
```
"""
import argparse

import joblib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from rlkit.state_distance.old.networks import NumpyGoalConditionedModelExtractor, \
    GoalConditionedDeltaModel, NumpyGoalConditionedDeltaModelExtractor, \
    TauBinaryGoalConditionedDeltaModel


def visualize_error_vs_tau(qf, policy, env, horizon):
    if (
        isinstance(qf, GoalConditionedDeltaModel)
        or isinstance(qf, TauBinaryGoalConditionedDeltaModel)
    ):
        model = NumpyGoalConditionedDeltaModelExtractor(qf)
    else:
        model = NumpyGoalConditionedModelExtractor(qf)
    actual_state = env.reset()

    tau_max = 10

    goal_state = env.sample_goal_for_rollout()
    policy.set_goal(goal_state)

    actual_states = []
    predicted_final_states = []
    taus = np.array(range(tau_max-1, -1, -1))
    for tau in taus:
        policy.set_tau(tau)
        actual_states.append(actual_state.copy())
        action, _ = policy.get_action(actual_state)
        predicted_final_states.append(
            model.next_state(actual_state, action, goal_state, tau)
        )
        actual_state = env.step(action)[0]
    final_state = actual_state

    predicted_final_states = np.array(predicted_final_states)
    errors = np.abs(final_state - predicted_final_states)
    distance_errors = np.abs(
        np.abs(
            env.convert_ob_to_goal(final_state)
            - goal_state
        ) - np.abs(
            env.convert_obs_to_goals(predicted_final_states) - goal_state
        )
    )
    num_state_dims = env.observation_space.low.size
    norm = colors.Normalize(vmin=0, vmax=num_state_dims)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)

    for data, name in [
        (errors, "State Abs Error"),
        (distance_errors, "State Distance Abs Error"),
    ]:
        plt.figure()
        for dim in range(data.shape[1]):
            plt.plot(
                taus,
                data[:, dim],
                '.-',
                label='Dim {}'.format(dim),
                color=mapper.to_rgba(dim),
            )
        plt.xlabel("Tau")
        plt.ylabel(name)
        plt.legend(loc='best')
        plt.show()


def visualize_accumulated_error(qf, policy, env, horizon):
    if (
        isinstance(qf, GoalConditionedDeltaModel)
        or isinstance(qf, TauBinaryGoalConditionedDeltaModel)
    ):
        model = NumpyGoalConditionedDeltaModelExtractor(qf)
    else:
        model = NumpyGoalConditionedModelExtractor(qf)
    # policy = UniformRandomPolicy(env.action_space)
    actual_state = env.reset()

    predicted_states = []
    actual_states = []
    goal_state = env.sample_goal_for_rollout()
    policy.set_goal(goal_state)
    policy.set_tau(0)

    predicted_state = actual_state
    for _ in range(horizon):
        predicted_states.append(predicted_state.copy())
        actual_states.append(actual_state.copy())

        action, _ = policy.get_action(actual_state)
        predicted_state = model.next_state(
            predicted_state, action, goal_state, 0
        )
        actual_state = env.step(action)[0]

    predicted_states = np.array(predicted_states)
    actual_states = np.array(actual_states)
    times = np.arange(horizon)

    num_state_dims = env.observation_space.low.size
    dims = list(range(num_state_dims))
    norm = colors.Normalize(vmin=0, vmax=num_state_dims)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
    plt.figure()
    for dim in dims:
        plt.plot(
            times,
            predicted_states[:, dim],
            '.-',
            label='Predicted, Dim {}'.format(dim),
            color=mapper.to_rgba(dim),
        )
        plt.plot(
            times,
            actual_states[:, dim],
            '.--',
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
    parser.add_argument('--tau', type=int, default=5, help='Horizon for eval')
    args = parser.parse_args()

    data = joblib.load(args.file)
    qf = data['qf']
    policy = data['policy']
    env = data['env']
    policy.train(False)
    qf.train(False)
    visualize_error_vs_tau(qf, policy, env, args.H)
    visualize_accumulated_error(qf, policy, env, args.H)
