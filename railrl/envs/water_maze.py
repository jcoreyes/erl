from collections import deque, OrderedDict

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from railrl.envs.env_utils import get_asset_xml
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces.box import Box
from rllab.misc import logger

RADIUS = 0.1


class WaterMaze(ProxyEnv, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())
        super().__init__(MujocoWaterMaze(**kwargs))

    def get_tf_loss(self, observations, actions, target_labels, **kwargs):
        """
        Return the supervised-learning loss.
        :param observation: Tensor
        :param action: Tensor
        :return: loss Tensor
        """
        return -(actions + observations - target_labels)**2

    def get_param_values(self):
        return None

    def log_diagnostics(self, paths, **kwargs):
        # import ipdb; ipdb.set_trace()
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        # env_infos = [p['env_infos'] for p in paths]
        # target_positions = [info['target_position'] for info in env_infos]
        # radius = [info['radius'] for info in env_infos]

        returns = []
        for path in paths:
            target_position = path["env_infos"]["target_position"]
            radius = path["env_infos"]["radius"]
            def compute_reward(obs, action):
                position = obs[:2]
                dist = np.linalg.norm(position - target_position)
                on_platform = dist <= radius
                return (
                    on_platform
                    - self._wrapped_env.l2_action_penalty_weight
                    * np.linalg.norm(action)
                )
            rewards = [compute_reward(obs, action)
                       for obs, action
                       in zip(path['observations'], path['actions'])]
            returns.append(np.sum(rewards))
        last_statistics = OrderedDict()
        last_statistics.update(create_stats_ordered_dict(
            'Return',
            returns,
        ))
        last_statistics.update(create_stats_ordered_dict(
            'Actions',
            actions,
        ))

        for key, value in last_statistics.items():
            logger.record_tabular(key, value)
        return rewards

    def terminate(self):
        self._wrapped_env.close()

    @staticmethod
    def get_extra_info_dict_from_batch(batch):
        return {}

    @staticmethod
    def get_flattened_extra_info_dict_from_subsequence_batch(batch):
        return {}

    @staticmethod
    def get_last_extra_info_dict_from_subsequence_batch(batch):
        return {}


class MujocoWaterMaze(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, horizon=200, l2_action_penalty_weight=1e-2, **kwargs):
        utils.EzPickle.__init__(self)
        self.l2_action_penalty_weight = l2_action_penalty_weight
        self.horizon = horizon
        self._t = 0
        self._on_platform_history = deque(maxlen=5)
        for _ in range(5):
            self._on_platform_history.append(False)

        mujoco_env.MujocoEnv.__init__(self, get_asset_xml('water_maze.xml'), 2)
        self.target_low = self.observation_space.low[2:]
        self.target_high = self.observation_space.high[2:]
        self.action_space = Box(self.action_space.low[:2],
                                self.action_space.high[:2])
        self.observation_space = Box(
            np.hstack((self.observation_space.low[:2], [0])),
            np.hstack((self.observation_space.high[:2], [1])),
        )

    def _step(self, force_actions):
        self._t += 1
        mujoco_action = np.hstack([force_actions, [0, 0]])
        self.do_simulation(mujoco_action, self.frame_skip)
        observation = self._get_observation()

        on_platform = observation[2]
        self._on_platform_history.append(on_platform)

        if all(self._on_platform_history):
            self.reset_ball_position()

        reward = (
            on_platform
            - self.l2_action_penalty_weight * np.linalg.norm(force_actions)
        )
        done = self._t >= self.horizon
        info = {
            'radius': RADIUS,
            'target_position': self._get_target_position(),
        }
        return observation, reward, done, info

    def reset_ball_position(self):
        new_ball_position = self.np_random.uniform(size=2, low=-0.2, high=0.2)
        target_position = self._get_target_position()
        qvel = np.zeros(self.model.nv)
        new_pos = np.hstack((new_ball_position, target_position))
        self.set_state(new_pos, qvel)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq,
                                                       low=-0.2, high=0.2)
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        self._t = 0
        return self._get_observation()

    def _get_observation(self):
        position = np.concatenate([self.model.data.qpos]).ravel()[:2]
        dist = np.linalg.norm(position - self._get_target_position())
        on_platform = dist <= RADIUS
        return np.hstack((position, [on_platform]))

    def _get_target_position(self):
        return np.concatenate([self.model.data.qpos]).ravel()[2:]

    def viewer_setup(self):
        v = self.viewer
        # v.cam.trackbodyid=0
        # v.cam.distance = v.model.stat.extent


def make_heat_map(eval_func, resolution=50):
    linspace = np.linspace(-0.3, 0.3, num=resolution)
    map = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            map[i, j] = eval_func(np.array([linspace[i], linspace[j]]))
    return map


def make_density_map(paths, resolution=50):
    linspace = np.linspace(-0.3, 0.3, num=resolution + 1)
    y = paths[:, 0]
    x = paths[:, 1]
    H, xedges, yedges = np.histogram2d(y, x, bins=(linspace, linspace))
    H = H.astype(np.float)
    H = H / np.max(H)
    return H


def plot_maps(old_combined=None, *heatmaps):
    import matplotlib.pyplot as plt
    combined = np.c_[heatmaps]
    if old_combined is not None:
        combined = np.r_[old_combined, combined]
    plt.figure()
    plt.imshow(combined, cmap='afmhot', interpolation='none')
    plt.show()
    return combined


if __name__ == "__main__":
    def evalfn(a):
        return np.linalg.norm(a - np.array([0, 0]))


    hm = make_heat_map(evalfn, resolution=50)
    paths = np.random.randn(5000, 2) * 0.1
    dm = make_density_map(paths, resolution=50)
    a = plot_maps(None, hm, dm)
    plot_maps(a, hm, dm)
