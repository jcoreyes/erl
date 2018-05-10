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


class SawyerPushAndReachXYEnv(MujocoEnv, Serializable, MultitaskEnv):
    INIT_BLOCK_LOW = np.array([-0.05, 0.55])
    INIT_BLOCK_HIGH = np.array([0.05, 0.65])
    PUCK_GOAL_LOW = INIT_BLOCK_LOW
    PUCK_GOAL_HIGH = INIT_BLOCK_HIGH
    HAND_GOAL_LOW = INIT_BLOCK_LOW
    HAND_GOAL_HIGH = INIT_BLOCK_HIGH
    FIXED_PUCK_GOAL = np.array([0.05, 0.6])
    FIXED_HAND_GOAL = np.array([-0.05, 0.6])

    def __init__(
            self,
            reward_info=None,
            frame_skip=50,
            pos_action_scale=2. / 100,
            randomize_goals=True,
    ):
        self.quick_init(locals())
        self.reward_info = reward_info
        self.randomize_goals = randomize_goals
        self._pos_action_scale = pos_action_scale
        self._goal_xyxy = self.sample_goal_xyxy()
        MultitaskEnv.__init__(self, distance_metric_order=2)
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)

        self.action_space = Box(
            np.array([-1, -1]),
            np.array([1, 1]),
        )
        self.observation_space = Box(
            np.array([-0.2, 0.5, -0.2, 0.5]),
            np.array([0.2, 0.7, 0.2, 0.7]),
        )
        self.goal_space = Box(
            self.observation_space.low,
            self.observation_space.high,
        )
        self.reset()
        self.reset_mocap_welds()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_push_and_reach_mocap.xml')

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
        mocap_delta_z = 0.06 - self.data.mocap_pos[0, 2]
        new_mocap_action = np.hstack((
            a,
            np.array([mocap_delta_z])
        ))
        self.mocap_set_action(new_mocap_action[:3] * self._pos_action_scale)
        u = np.zeros(7)
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs, u, obs, self._goal_xyxy)
        done = False

        hand_distance = np.linalg.norm(
            self.get_hand_goal_pos() - self.get_endeff_pos()
        )
        puck_distance = np.linalg.norm(
            self.get_puck_goal_pos() - self.get_puck_pos())
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_puck_pos())
        info = dict(
            hand_distance=hand_distance,
            puck_distance=puck_distance,
            touch_distance=touch_distance,
            success=float(puck_distance < 0.03),
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
        e = self.get_endeff_pos()[:2]
        b = self.get_puck_pos()[:2]
        return np.concatenate((e, b))

    def get_puck_pos(self):
        return self.data.body_xpos[self.puck_id].copy()

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def get_hand_goal_pos(self):
        return self.data.body_xpos[self.hand_goal_id].copy()

    def get_puck_goal_pos(self):
        return self.data.body_xpos[self.puck_goal_id].copy()

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    @property
    def puck_id(self):
        return self.model.body_names.index('puck')

    @property
    def puck_goal_id(self):
        return self.model.body_names.index('puck-goal')

    @property
    def hand_goal_id(self):
        return self.model.body_names.index('hand-goal')

    def sample_goal_xyxy(self):
        if self.randomize_goals:
            hand = np.random.uniform(self.HAND_GOAL_LOW, self.HAND_GOAL_HIGH)
            puck = np.random.uniform(self.PUCK_GOAL_LOW, self.PUCK_GOAL_HIGH)
        else:
            hand = self.FIXED_HAND_GOAL.copy()
            puck = self.FIXED_PUCK_GOAL.copy()
        return np.hstack((hand, puck))

    def sample_puck_xy(self):
        pos = np.random.uniform(self.INIT_BLOCK_LOW, self.INIT_BLOCK_HIGH)
        while np.linalg.norm(self.get_endeff_pos()[:2] - pos) < 0.035:
            pos = np.random.uniform(self.INIT_BLOCK_LOW, self.INIT_BLOCK_HIGH)
        return pos

    def set_puck_xy(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[7:10] = np.hstack((pos.copy(), np.array([0.02])))
        qvel[7:10] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def set_goal_xyxy(self, xyxy):
        self._goal_xyxy = xyxy
        hand_goal = xyxy[:2]
        puck_goal = xyxy[-2:]
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[14:17] = np.hstack((hand_goal.copy(), np.array([0.02])))
        qvel[14:17] = [0, 0, 0]
        qpos[21:24] = np.hstack((puck_goal.copy(), np.array([0.02])))
        qvel[21:24] = [0, 0, 0]
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

    def reset(self):
        velocities = self.data.qvel.copy()
        angles = np.array(self.init_angles)
        self.set_state(angles.flatten(), velocities.flatten())
        for _ in range(10):
            new_mocap_pos = np.array([0, 0.6, 0.02])
            self.data.set_mocap_pos('mocap', new_mocap_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        # set_state resets the goal xy, so we need to explicit set it again
        self.set_goal_xyxy(self._goal_xyxy)
        self.set_puck_xy(self.sample_puck_xy())
        self.reset_mocap_welds()
        return self._get_obs()

    def compute_reward(self, ob, action, next_ob, goal):
        hand_xy = next_ob[:2]
        puck_xy = next_ob[-2:]
        hand_goal_xy = goal[:2]
        puck_goal_xy = goal[-2:]
        hand_dist = np.linalg.norm(hand_xy - hand_goal_xy)
        puck_dist = np.linalg.norm(puck_xy - puck_goal_xy)
        if not self.reward_info or self.reward_info["type"] == "euclidean":
            r = - hand_dist - puck_dist
        elif self.reward_info["type"] == "hand_only":
            r = - hand_dist
        elif self.reward_info["type"] == "puck_only":
            r = - puck_dist
        elif self.reward_info["type"] == "sparse":
            t = self.reward_info["threshold"]
            r = float(
                hand_dist + puck_dist < t
            ) - 1
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def compute_her_reward_np(self, ob, action, next_ob, goal):
        return self.compute_reward(ob, action, next_ob, goal)

    @property
    def init_angles(self):
        return [
            1.06139477e+00, -6.93988797e-01, 3.76729934e-01, 1.78410587e+00,
            - 5.36763074e-01, 5.88122189e-01, 3.51531533e+00,
            0.05, 0.55, 0.02,
            1, 0, 0, 0,
            0, 0.6, 0.02,
            1, 0, 1, 0,
            0, 0.6, 0.02,
            1, 0, 1, 0,
        ]

    def log_diagnostics(self, paths, logger=logger, prefix=""):
        super().log_diagnostics(paths)

        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'puck_distance',
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
        return 4

    def sample_goal_for_rollout(self):
        return self.sample_goal_xyxy()

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        self.set_goal_xyxy(goal)

    def convert_obs_to_goals(self, obs):
        return obs

    def sample_goals(self, batch_size):
        raise NotImplementedError()
