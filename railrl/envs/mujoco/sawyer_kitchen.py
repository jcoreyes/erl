import abc
import mujoco_py
import numpy as np
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

from railrl.core.serializable import Serializable
from railrl.envs.env_utils import get_asset_full_path
from railrl.envs.multitask.multitask_env import MultitaskEnv


class SawyerKitchenEnv(MujocoEnv, Serializable, metaclass=abc.ABCMeta):
    def __init__(self, frame_skip=30):
        self.quick_init(locals())
        MujocoEnv.__init__(self, self.model_path, frame_skip=frame_skip)

        self.observation_space = Box(
            np.array([-0.2, 0.5, 0]),
            np.array([0.2, 0.7, 0.5]),
        )
        self.goal_space = Box(
            np.array([-0.2, 0.5, 0]),
            np.array([0.2, 0.7, 0.5]),
        )
        self.reset()
        self.reset_mocap_welds()

    @property
    def model_path(self):
        return get_asset_full_path('kitchen/sawyer_kitchen.xml')

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.mocap_set_action(a[:3] / 100)
        u = np.zeros((8))
        u[7] = a[3]
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()
        reward = self._compute_reward(u, obs)
        done = False

        info = dict(
            cabinet_angle=self.get_cabinet_angle(),
        )
        return obs, reward, done, info

    def mocap_set_action(self, action):
        pos_delta = action[None]
        # self.reset_mocap2body_xpos()
        self.data.set_mocap_pos('mocap', self.data.mocap_pos + pos_delta)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def reset_mocap2body_xpos(self):
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.body_xpos[self.endeff_id]]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.body_xquat[self.endeff_id]]),
        )

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def get_cabinet_angle(self):
        return np.array([self.data.get_joint_qpos('doorjoint')])

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self, action, next_obs):
        raise NotImplementedError()

    def reset(self):
        # angles = self.data.qpos.copy()
        # velocities = self.data.qvel.copy()
        # angles[:] = self.init_angles
        # self.set_state(angles.flatten(), velocities.flatten())
        return self._get_obs()


class KitchenCabinetEnv(SawyerKitchenEnv, MultitaskEnv):
    def __init__(self, **kwargs):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        super().__init__(**kwargs)
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.observation_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.goal_space = Box(
            np.array([-1]),
            np.array([1]),
        )

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_cabinet_angle()
        return np.concatenate((e, b))

    @property
    def goal_dim(self) -> int:
        return 1

    def sample_goals(self, batch_size):
        return np.random.rand(batch_size) * 2 - 1

    def convert_obs_to_goals(self, obs):
        """
        Convert a raw environment observation into a goal (if possible).
        """
        return obs[:, -1:]

    def _compute_reward(self, action, next_obs):
        actual_angle = self.convert_ob_to_goal(next_obs)
        return - np.abs(actual_angle - self.multitask_goal)
