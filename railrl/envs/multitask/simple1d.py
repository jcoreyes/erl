"""
Simple 1-D environments for debugging
"""
import numpy as np
import matplotlib.pyplot as plt
from gym import Env
from gym.spaces import Box, Discrete

from railrl.envs.multitask.multitask_env import MultitaskEnv
from rllab.core.serializable import Serializable


class Simple1D(MultitaskEnv, Env, Serializable):
    metadata = {'render.modes': ['human']}

    def __init__(self, distance_metric_order=1):
        Serializable.quick_init(self, locals())
        super().__init__(distance_metric_order=distance_metric_order)
        self._state = 0
        self.action_space = Box(-1, 1, shape=(1,))
        self.observation_space = Box(-10, 10, shape=(1,))

    def convert_obs_to_goals(self, obs):
        return obs

    @property
    def goal_dim(self) -> int:
        return 1

    def sample_goals(self, batch_size):
        return np.random.uniform(-10, 10, (batch_size, 1))

    def _reset(self):
        self._state = 0

    def _step(self, action):
        self._state += action
        self._state = np.clip(self._state, -10, 10)
        observation = self._state
        reward = -np.abs(self._state - self.multitask_goal)
        done = False
        info = {}
        return observation, reward, done, info

    def _render(self, mode='human', close=False):
        board = [' '] * 21
        goal_pos = int(self.multitask_goal + 10)
        board[goal_pos] = 'x'
        current_pos = int(self._state + 10)
        board[current_pos] = str(int((self._state % 1) * 10))
        str_repr = "[{}]".format(''.join(board))
        print(str_repr)

    def _seed(self, seed=None):
        return []


class DiscreteSimple1D(Simple1D):
    def __init__(self, num_bins=5, distance_metric_order=1):
        Serializable.quick_init(self, locals())
        super().__init__(distance_metric_order=distance_metric_order)
        self.num_bins = num_bins
        self.idx_to_continuous_action = np.linspace(-1, 1, num_bins)
        self.action_space = Discrete(len(self.idx_to_continuous_action))

    def _step(self, a):
        continuous_action = self.idx_to_continuous_action[a]
        return super()._step(continuous_action)


class Simple1DTDMPlotter(object):
    def __init__(self, tdm, policy, obs_lst, default_action, n_samples):
        self._tdm = tdm
        self._policy = policy
        self._obs_lst = obs_lst
        self._default_action = default_action
        self._n_samples = n_samples

        self._var_inds = np.where(np.isnan(default_action))[0]
        assert len(self._var_inds) == 2

        n_plots = len(obs_lst)

        x_size = 5 * n_plots
        y_size = 5

        fig = plt.figure(figsize=(x_size, y_size))
        self._ax_lst = []
        for i in range(n_plots):
            ax = fig.add_subplot(100 + n_plots * 10 + i + 1)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            ax.grid(True)
            self._ax_lst.append(ax)

        self._line_objects = list()

    def draw(self):
        # noinspection PyArgumentList
        [h.remove() for h in self._line_objects]
        self._line_objects = list()

        self._plot_level_curves()
        self._plot_action_samples()

        plt.draw()
        plt.pause(0.001)

    def _plot_level_curves(self):
        # Create mesh grid.
        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)
        xgrid, ygrid = np.meshgrid(xs, ys)
        N = len(xs)*len(ys)

        # Copy default values along the first axis and replace nans with
        # the mesh grid points.
        actions = np.tile(self._default_action, (N, 1))
        actions[:, self._var_inds[0]] = xgrid.ravel()
        actions[:, self._var_inds[1]] = ygrid.ravel()

        for ax, obs in zip(self._ax_lst, self._obs_lst):
            repeated_obs = np.repeat(
                obs[None],
                actions.shape[0],
                axis=0,
            )
            qs = self._qf.eval_np(repeated_obs, actions)
            qs = qs.reshape(xgrid.shape)

            cs = ax.contour(xgrid, ygrid, qs, 20)
            self._line_objects += cs.collections
            self._line_objects += ax.clabel(
                cs, inline=1, fontsize=10, fmt='%.2f')

    def _plot_action_samples(self):
        for ax, obs in zip(self._ax_lst, self._obs_lst):
            actions = self._policy.get_actions(
                np.ones((self._n_samples, 1)) * obs[None, :])

            x, y = actions[:, 0], actions[:, 1]
            self._line_objects += ax.plot(x, y, 'b*')
