import numpy as np
from torch.autograd import Variable

from railrl.torch import pytorch_util as ptu


class GreedyModelBasedPolicy(object):
    """
    Choose action according to

    a = argmin_a ||f(s, a) - GOAL||^2

    where f is a learned forward dynamics model.

    Do the argmax by sampling a bunch of acitons
    """
    def __init__(
            self,
            model,
            env,
            sample_size=100,
    ):
        self.model = model
        self.env = env
        self.sample_size = sample_size
        self._goal_batch = None

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.sample_size,
            axis=0
        )
        return Variable(
            ptu.from_numpy(array_expanded).float(),
            requires_grad=False,
        )

    def set_goal(self, goal):
        self._goal_batch = self.expand_np_to_var(goal)

    def reset(self):
        pass

    def get_action(self, obs):
        sampled_actions = self.env.sample_actions(self.sample_size)
        action = ptu.np_to_var(sampled_actions)
        obs = self.expand_np_to_var(obs)
        state_delta_predicted = self.model(
            obs,
            action,
        )
        next_state_predicted = obs + state_delta_predicted
        next_goal_state_predicted = ptu.np_to_var(
            self.env.convert_obs_to_goal_states(
                ptu.get_numpy(
                    next_state_predicted
                )
            )
        )
        errors = (next_goal_state_predicted - self._goal_batch)**2
        mean_errors = errors.mean(dim=1)
        min_i = np.argmin(ptu.get_numpy(mean_errors))
        return sampled_actions[min_i], {}