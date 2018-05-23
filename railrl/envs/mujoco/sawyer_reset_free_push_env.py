from collections import OrderedDict
import numpy as np
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

from railrl.core import logger

from railrl.core.serializable import Serializable
from railrl.envs.env_utils import get_asset_full_path
from railrl.envs.mujoco.sawyer_base import SawyerMocapBase
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths


class SawyerResetFreePushEnv(SawyerMocapBase, MujocoEnv, Serializable,
                             MultitaskEnv):
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
            np.array([-1, -1]),
            np.array([1, 1]),
        )

        self.observation_space = Box(
            np.array([-0.2, 0.5, 0, -0.2, 0.5]),
            np.array([0.2, 0.7, 0.5, 0.2, 0.7]),
        )
        self.goal_space = Box(
            np.array([-0.2, 0.5]),
            np.array([0.2, 0.7]),
        )
        velocities = self.data.qvel.copy()
        angles = np.array(self.init_angles)
        self.set_state(angles.flatten(), velocities.flatten())
        self.reset()
        self.reset_mocap_welds()

    # def viewer_setup(self):
    #     SawyerMocapBase.viewer_setup(self)

    @property
    def model_name(self):
        if self.hide_goal:
            return get_asset_full_path('sawyer_push_joint_limited_mocap_goal_hidden.xml')
        else:
            return get_asset_full_path('sawyer_push_joint_limited_mocap.xml')

    def step(self, a):
        a = np.clip(a[:2], -1, 1)
        mocap_delta_z = 0.06 - self.data.mocap_pos[0, 2]
        new_mocap_action = np.hstack((
            a,
            np.array([mocap_delta_z])
        ))
        self.mocap_set_action(new_mocap_action * self._pos_action_scale)
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

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_block_pos()[:2]
        return np.concatenate((e, b))

    def sample_goal_xy(self):
        if self.randomize_goals:
            pos = np.random.uniform(self.INIT_GOAL_LOW, self.INIT_GOAL_HIGH)
        else:
            pos = self.FIXED_GOAL_INIT.copy()
        return pos

    def sample_block_xy(self):
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
        qpos[9:12] = np.hstack((pos.copy(), np.array([0.02])))
        qvel[9:12] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def reset(self):
        self.set_goal_xy(self._goal_xy)
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

    @property
    def init_angles(self):
        return [1.78026069e+00, - 6.84415781e-01, - 1.54549231e-01,
                2.30672090e+00, 1.93111471e+00,  1.27854012e-01,
                1.49353907e+00,
                1.80196716e-03, 7.40415706e-01,
                - 3.62518873e-02, 6.13435141e-01, 2.09686080e-02,
                7.07106781e-01, 1.48979724e-14, 7.07106781e-01, - 1.48999170e-14
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

    def sample_hand_xy(self):
        return np.random.uniform(
            [-0.1, -0.1 + 0.6],
            [0.1, 0.1 + 0.6],
            size=2
        )

    def set_hand_xy(self, xy):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([xy[0], xy[1], 0.02]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)

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

    def set_to_goal(self, goal):
        self.set_block_xy(goal)

    def convert_obs_to_goals(self, obs):
        return obs[:, -2:]

    def sample_goals(self, batch_size):
        return np.random.uniform(
            self.INIT_GOAL_LOW,
            self.INIT_GOAL_HIGH,
            size=(batch_size, 2),
        )
