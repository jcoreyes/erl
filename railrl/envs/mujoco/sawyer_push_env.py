from collections import OrderedDict
import numpy as np
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import mujoco_py

from railrl.core import logger

from railrl.core.serializable import Serializable
from railrl.envs.env_utils import get_asset_full_path
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths


class SawyerPushEnv(MujocoEnv, Serializable, MultitaskEnv):
    """Implements a 3D-position controlled Sawyer environment"""
    INIT_BLOCK_LOW = np.array([-0.05, 0.55])
    INIT_BLOCK_HIGH = np.array([0.05, 0.65])
    INIT_GOAL_LOW = INIT_BLOCK_LOW
    INIT_GOAL_HIGH = INIT_BLOCK_HIGH
    FIXED_GOAL_INIT = np.array([0.05, 0.6])
    INIT_HAND_POS = np.array([0, 0.4, 0.02])

    def __init__(
            self,
            reward_info=None,
            frame_skip=50,
            pos_action_scale=2. / 100,
            randomize_goals=True,
            hide_goal=False,
    ):
        self.quick_init(locals())
        self.reward_info = reward_info
        self.randomize_goals = randomize_goals
        self._pos_action_scale = pos_action_scale
        self.hide_goal = hide_goal
        self._goal_xy = self.sample_goal_xy()
        MultitaskEnv.__init__(self, distance_metric_order=2)
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)

        self.action_space = Box(
            np.array([-1, -1, -1]),
            np.array([1, 1, 1]),
        )
        self.observation_space = Box(
            np.array([-0.2, 0.5, 0, -0.2, 0.5]),
            np.array([0.2, 0.7, 0.5, 0.2, 0.7]),
        )
        self.goal_space = Box(
            np.array([-0.2, 0.5]),
            np.array([0.2, 0.7]),
        )
        self.reset()
        self.reset_mocap_welds()

    @property
    def model_name(self):
        if self.hide_goal:
            return get_asset_full_path('sawyer_push_mocap_goal_hidden.xml')
        else:
            return get_asset_full_path('sawyer_push_mocap.xml')

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0

        # robot view
        # rotation_angle = 90
        # cam_dist = 1
        # cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])

        # 3rd person view
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

        # top down view
        # cam_dist = 0.2
        # rotation_angle = 0
        # cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])

        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.mocap_set_action(a[:3] * self._pos_action_scale)
        u = np.zeros(7)
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs, u, obs, self._goal_xy)
        done = False

        hand_to_goal_dist = np.linalg.norm(
            self.get_goal_pos() - self.get_endeff_pos()
        )
        block_distance = np.linalg.norm(
            self.get_goal_pos() - self.get_block_pos())
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_block_pos())
        info = dict(
            hand_to_goal_dist=hand_to_goal_dist,
            block_distance=block_distance,
            touch_distance=touch_distance,
            success=float(block_distance < 0.03),
        )
        return obs, reward, done, info

    def mocap_set_action(self, action):
        pos_delta = action[None]
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, 0] = np.clip(
            new_mocap_pos[0, 0],
            -0.1,
            0.1,
        )
        new_mocap_pos[0, 1] = np.clip(
            new_mocap_pos[0, 1],
            -0.1 + 0.6,
            0.1 + 0.6,
            )
        new_mocap_pos[0, 2] = np.clip(
            new_mocap_pos[0, 2],
            0,
            0.5,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_block_pos()[:2]
        return np.concatenate((e, b))

    def get_block_pos(self):
        return self.data.body_xpos[self.block_id].copy()

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def get_goal_pos(self):
        return self.data.body_xpos[self.goal_id].copy()

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    @property
    def block_id(self):
        return self.model.body_names.index('block')

    @property
    def goal_id(self):
        return self.model.body_names.index('goal')

    def sample_goal_xy(self):
        if self.randomize_goals:
            pos = np.random.uniform(self.INIT_GOAL_LOW, self.INIT_GOAL_HIGH)
        else:
            pos = self.FIXED_GOAL_INIT.copy()
        return pos

    def sample_block_xy(self):
        raise NotImplementedError("Shouldn't you use SawyerPushXYEasyEnv? Ask Vitchyr")
        pos = np.random.uniform(self.INIT_BLOCK_LOW, self.INIT_BLOCK_HIGH)
        while np.linalg.norm(self.get_endeff_pos()[:2] - pos) < 0.035:
            pos = np.random.uniform(self.INIT_BLOCK_LOW, self.INIT_BLOCK_HIGH)
        return pos

    def set_block_xy(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[7:10] = np.hstack((pos.copy(), np.array([0.02])))
        qvel[7:10] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def set_goal_xy(self, pos):
        self._goal_xy = pos.copy()
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[14:17] = np.hstack((pos.copy(), np.array([0.02])))
        qvel[14:17] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()

    def reset_mocap2body_xpos(self):
        # move mocap to weld joint
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.body_xpos[self.endeff_id]]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.body_xquat[self.endeff_id]]),
        )

    def reset(self, resample_block=True):
        velocities = self.data.qvel.copy()
        angles = np.array(self.init_angles)
        self.set_state(angles.flatten(), velocities.flatten())
        init_block_pos = self.get_block_pos()
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.INIT_HAND_POS)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        # set_state resets the goal xy, so we need to explicit set it again
        self.set_goal_xy(self._goal_xy)
        if resample_block:
            self.set_block_xy(self.sample_block_xy())
        else:
            self.set_block_xy(init_block_pos[:2])
        self.reset_mocap_welds()
        return self._get_obs()

    def compute_reward(self, ob, action, next_ob, goal):
        hand_xy = next_ob[:2]
        obj_xy = next_ob[-2:]
        if not self.reward_info or self.reward_info["type"] == "euclidean":
            r = - np.linalg.norm(obj_xy - goal)
        elif self.reward_info["type"] == "shaped":
            r = - np.linalg.norm(obj_xy - goal) - np.linalg.norm(
                hand_xy - obj_xy
            )
        elif self.reward_info["type"] == "hand_to_object_only":
            r = - np.linalg.norm(hand_xy - obj_xy)
        elif self.reward_info["type"] == "sparse":
            t = self.reward_info["threshold"]
            r = float(
                np.linalg.norm(obj_xy - goal) < t
            ) - 1
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def compute_her_reward_np(self, ob, action, next_ob, goal):
        return self.compute_reward(ob, action, next_ob, goal)

    # @property
    # def init_angles(self):
    #     return [
    #         1.06139477e+00, -6.93988797e-01, 3.76729934e-01, 1.78410587e+00,
    #         - 5.36763074e-01, 5.88122189e-01, 3.51531533e+00,
    #         0.05, 0.55, 0.02,
    #         1, 0, 0, 0,
    #         0, 0.6, 0.02,
    #         1, 0, 1, 0,
    #     ]

    @property
    def init_angles(self):
        return [1.78026069e+00, - 6.84415781e-01, - 1.54549231e-01,
                2.30672090e+00, 1.93111471e+00,  1.27854012e-01,
                1.49353907e+00, 1.80196716e-03, 7.40415706e-01,
                2.09895360e-02,  9.99999990e-01,  3.05766105e-05,
                - 3.78462492e-06, 1.38684523e-04, - 3.62518873e-02,
                6.13435141e-01, 2.09686080e-02,  7.07106781e-01,
                1.48979724e-14, 7.07106781e-01, - 1.48999170e-14
                ]

    def log_diagnostics(self, paths, logger=logger, prefix=""):
        super().log_diagnostics(paths)

        statistics = OrderedDict()
        for stat_name in [
            'hand_to_goal_dist',
            'block_distance',
            'touch_distance',
            'success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s %s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s %s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    """
    Multitask functions
    """

    @property
    def goal_dim(self) -> int:
        return 2

    def sample_goal_for_rollout(self):
        return self.sample_goal_xy()

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        self.set_goal_xy(goal)
        self.set_to_goal(goal)

    def set_to_goal(self, goal):
        # Hack for now since there isn't a goal hand position
        self.reset(resample_block=False)
        self.set_block_xy(goal)

    def convert_obs_to_goals(self, obs):
        return obs[:, -2:]

    def sample_goals(self, batch_size):
        return np.random.uniform(
            self.INIT_GOAL_LOW,
            self.INIT_GOAL_HIGH,
            size=(batch_size, 2),
        )


class SawyerPushXYEnv(SawyerPushEnv):
    """
    Only move along XY-axis.
    """
    def __init__(self, **kwargs):
        self.quick_init(locals())
        super().__init__(**kwargs)

        self.action_space = Box(
            np.array([-1, -1]),
            np.array([1, 1]),
        )

    def step(self, a):
        a = np.clip(a, -1, 1)
        mocap_delta_z = 0.06 - self.data.mocap_pos[0, 2]
        new_mocap_action = np.hstack((
            a,
            np.array([mocap_delta_z])
        ))
        return super().step(new_mocap_action)


class SawyerPushXYEasyEnv(SawyerPushXYEnv):
    """
    Always start the block in the same position
    """
    INIT_GOAL_LOW = np.array([-0.05, 0.6])
    INIT_GOAL_HIGH = np.array([0.05, 0.75])

    def sample_block_xy(self):
        return np.array([0, 0.6])


if __name__ == "__main__":
    e = SawyerPushEnv(reward_info=dict(type="euclidean"))
    for j in range(50):
        e.reset()
        for i in range(8):
            desired = np.random.uniform(
                low=(-0.01, -0.01, -0.01, -1),
                high=(0.01, 0.01, 0.01, 1)
            )
            print(i)
            for k in range(200):
                e.step(desired)

                e.render()
