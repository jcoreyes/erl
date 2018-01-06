"""
Simple 1-D environments for debugging
"""
import numpy as np
import matplotlib.pyplot as plt
from gym import Env
from gym.spaces import Box, Discrete

from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.state_distance.util import merge_into_flat_obs
from railrl.misc.visualization_util import plot_heatmap, HeatMap
from railrl.core.serializable import Serializable
from railrl.core import logger


class Simple1D(MultitaskEnv, Env, Serializable):
    metadata = {'render.modes': ['human']}

    def __init__(self, distance_metric_order=1):
        Serializable.quick_init(self, locals())
        super().__init__(distance_metric_order=distance_metric_order)
        self._state = np.zeros(1)
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
        self._state = np.zeros(1)
        return self._state

    def _step(self, action):
        # Do not do += since that will use the same memory
        self._state = self._state + action
        self._state = np.clip(self._state, -10, 10)
        observation = self._state
        reward = -float(np.abs(self._state - self.multitask_goal))
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


ACTION_LIMITS = (-1, 1)


class Simple1DTdmPlotter(object):
    def __init__(self, tdm, location_lst, goal_lst, max_tau, grid_size=20):
        self._tdm = tdm
        self.max_tau_evaluated = max_tau + 1
        self.grid_size = grid_size

        x_size = 2.5 * len(goal_lst)
        y_size = 2.5 * len(location_lst)

        self.fig = plt.figure(figsize=(x_size, y_size))
        self._ax_lst = []
        self._img_lst = []
        i = 1
        self._location_lst = []
        self._goal_lst = []
        for row, goal in enumerate(goal_lst):
            for col, location in enumerate(location_lst):
                ax = self.fig.add_subplot(len(location_lst), len(goal_lst), i)
                i += 1
                ax.set_xlim(ACTION_LIMITS)
                ax.set_ylim((0, self.max_tau_evaluated))
                ax.grid(True)
                ax.set_xlabel("Action")
                ax.set_ylabel("Tau")
                ax.set_title("X = {}, Goal = {}".format(
                    float(location), float(goal)
                ))
                self._img_lst.append(None)
                self._ax_lst.append(ax)
                self._goal_lst.append(goal)
                self._location_lst.append(location)
        self.fig.subplots_adjust(hspace=0.5, wspace=0.5)

        self._line_objects = list()

    def draw(self):
        # noinspection PyArgumentList
        [h.remove() for h in self._line_objects]
        self._line_objects = list()

        self._plot_level_curves()

        plt.draw()
        plt.savefig(logger.get_snapshot_dir() + "/tdm_vis.png")
        plt.pause(0.001)

    def _plot_level_curves(self):
        # Create mesh grid.
        actions_sampled = np.linspace(
            ACTION_LIMITS[0], ACTION_LIMITS[1], self.grid_size
        )
        taus_sampled = np.linspace(
            0, self.max_tau_evaluated, 2*self.max_tau_evaluated
        )
        # Make it so that when rounded, the last value will be max_tau
        # This really only matters when (e.g.) the the taus are converted to
        # one-hot vectors
        taus_sampled = np.clip(taus_sampled, 0, self.max_tau_evaluated - 1e-6)
        # action = x axis
        action_grid, tau_grid = np.meshgrid(actions_sampled, taus_sampled)
        N = len(actions_sampled)*len(taus_sampled)

        actions = action_grid.ravel()
        taus = tau_grid.ravel()
        actions = np.expand_dims(actions, 1)
        taus = np.expand_dims(taus, 1)
        for i, (ax, obs, goal) in enumerate(zip(
                self._ax_lst, self._location_lst, self._goal_lst
        )):
            repeated_obs = np.repeat(np.array([[obs]]), N, axis=0)
            repeated_goals = np.repeat(np.array([[goal]]), N, axis=0)
            new_obs = merge_into_flat_obs(
                obs=repeated_obs,
                goals=repeated_goals,
                num_steps_left=taus,
            )
            qs = self._tdm.eval_np(new_obs, actions)
            q_grid = qs.reshape(action_grid.shape)
            img = self._img_lst[i]
            if img is None:
                hm = HeatMap(
                    q_grid,
                    actions_sampled,
                    taus_sampled,
                    {},
                )
                img, side_axis = plot_heatmap(self.fig, ax, hm)
                self._img_lst[i] = img
            img.set_data(q_grid)


class Simple1DTdmDiscretePlotter(object):
    def __init__(self, tdm, location_lst, goal_lst, max_tau, grid_size=20):
        self._tdm = tdm
        self.max_tau_evaluated = max_tau + 1
        self.grid_size = grid_size

        x_size = 2.5 * len(goal_lst)
        y_size = 2.5 * len(location_lst)

        self.fig = plt.figure(figsize=(x_size, y_size))
        self._ax_lst = []
        self._img_lst = []
        i = 1
        self._location_lst = []
        self._goal_lst = []
        for row, goal in enumerate(goal_lst):
            for col, location in enumerate(location_lst):
                ax = self.fig.add_subplot(len(location_lst), len(goal_lst), i)
                i += 1
                ax.set_xlim(ACTION_LIMITS)
                ax.set_ylim((0, self.max_tau_evaluated))
                ax.grid(True)
                ax.set_xlabel("Action")
                ax.set_ylabel("Tau")
                ax.set_title("X = {}, Goal = {}".format(
                    float(location), float(goal)
                ))
                self._img_lst.append(None)
                self._ax_lst.append(ax)
                self._goal_lst.append(goal)
                self._location_lst.append(location)

        self._line_objects = list()

    def draw(self):
        # noinspection PyArgumentList
        [h.remove() for h in self._line_objects]
        self._line_objects = list()

        self._plot_level_curves()

        plt.draw()
        plt.savefig(logger.get_snapshot_dir() + "/tdm_vis.png")
        plt.pause(0.001)

    def _plot_level_curves(self):
        # Create mesh grid.
        actions_sampled = np.linspace(
            ACTION_LIMITS[0], ACTION_LIMITS[1], self.grid_size
        )
        taus_sampled = np.linspace(
            0, self.max_tau_evaluated, 2*self.max_tau_evaluated
        )
        # Make it so that when rounded, the last value will be max_tau
        # This really only matters when (e.g.) the the taus are converted to
        # one-hot vectors
        taus_sampled = np.clip(taus_sampled, 0, self.max_tau_evaluated - 1e-6)
        # action = x axis
        action_grid, tau_grid = np.meshgrid(actions_sampled, taus_sampled)
        N = len(actions_sampled)*len(taus_sampled)

        actions = action_grid.ravel()
        taus = tau_grid.ravel()
        actions = np.expand_dims(actions, 1)
        taus = np.expand_dims(taus, 1)
        for i, (ax, obs, goal) in enumerate(zip(
                self._ax_lst, self._location_lst, self._goal_lst
        )):
            repeated_obs = np.repeat(np.array([[obs]]), N, axis=0)
            repeated_goals = np.repeat(np.array([[goal]]), N, axis=0)
            new_obs = merge_into_flat_obs(
                obs=repeated_obs,
                goals=repeated_goals,
                num_steps_left=taus,
            )
            qs = self._tdm.eval_np(new_obs)
            q_grid = qs.reshape(action_grid.shape)
            img = self._img_lst[i]
            if img is None:
                hm = HeatMap(
                    q_grid,
                    actions_sampled,
                    taus_sampled,
                    {},
                )
                img, side_axis = plot_heatmap(self.fig, ax, hm)
                self._img_lst[i] = img
            img.set_data(q_grid)
