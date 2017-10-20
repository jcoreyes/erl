import numpy as np

from railrl.algos.state_distance.state_distance_q_learning import (
    StateDistanceQLearning
)


class VectorizedSdql(StateDistanceQLearning):
    def compute_rewards(self, obs, actions, next_obs, goal_states):
        return np.abs(next_obs - goal_states)
