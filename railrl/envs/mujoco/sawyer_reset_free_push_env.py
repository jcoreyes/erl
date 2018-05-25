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
    PUCK_GOAL_LOW = np.array([-0.05, 0.55])
    PUCK_GOAL_HIGH = np.array([0.05, 0.65])
    PUCK_GOAL_LARGE_LOW = np.array([-0.1, 0.5])
    PUCK_GOAL_LARGE_HIGH = np.array([0.1, 0.65])
    HAND_GOAL_LOW = np.array([-0.1, 0.5])
    HAND_GOAL_HIGH = np.array([0.1, 0.7])
    FIXED_GOAL_INIT = np.array([0.05, 0.6])
    INIT_HAND_POS = np.array([0, 0.4, 0.02])

    def __init__(
            self,
            reward_info=None,
            frame_skip=50,
            pos_action_scale=2. / 100,
            randomize_goals=True,
            hide_goal=False,
            puck_limit='normal'
    ):
        self.quick_init(locals())
        self.reward_info = reward_info
        self.randomize_goals = randomize_goals
        self._pos_action_scale = pos_action_scale
        self.hide_goal = hide_goal
        self.puck_limit = puck_limit
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
        velocities = self.data.qvel.copy()
        angles = np.array(self.init_angles)
        self.set_state(angles.flatten(), velocities.flatten())
        self.reset()
        self.reset_mocap_welds()

    @property
    def model_name(self):
        if self.puck_limit == 'normal':
            if self.hide_goal:
                return get_asset_full_path('sawyer_push_joint_limited_mocap_goal_hidden.xml')
            else:
                return get_asset_full_path('sawyer_push_joint_limited_mocap.xml')
        elif self.puck_limit == 'large':
            if self.hide_goal:
                return get_asset_full_path(
                    'sawyer_push_joint_limited_large_mocap_goal_hidden.xml'
                )
            else:
                return get_asset_full_path(
                    'sawyer_push_joint_limited_large_mocap.xml'
                )
        else:
            raise ValueError("Unrecoginzed puck limit: {}".format(
                self.puck_limit
            ))

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
        reward = self.compute_her_reward_np(obs, u, obs, self._goal_xyxy)
        done = False

        hand_to_goal_dist = np.linalg.norm(
            self.get_hand_goal_pos() - self.get_endeff_pos()
        )
        puck_distance = np.linalg.norm(
            self.get_puck_goal_pos() - self.get_puck_pos())
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_puck_pos())
        sum_distance = puck_distance + touch_distance
        info = dict(
            hand_to_goal_dist=hand_to_goal_dist,
            puck_distance=puck_distance,
            touch_distance=touch_distance,
            sum_distance=sum_distance,
            success=float(sum_distance < 0.1),
        )
        return obs, reward, done, info

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_puck_pos()[:2]
        return np.concatenate((e, b))

    def reset(self):
        self.set_goal_xyxy(self._goal_xyxy)
        self.reset_mocap_welds()
        return self._get_obs()

    def compute_her_reward_np(self, ob, action, next_ob, goal, env_info=None):
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

    @property
    def init_angles(self):
        return [
            # joints
            1.78026069e+00, - 6.84415781e-01, - 1.54549231e-01,
            2.30672090e+00, 1.93111471e+00,  1.27854012e-01,
            1.49353907e+00,
            # puck xy
            0, 0.7,
            # hand goal
            - 3.62518873e-02, 6.13435141e-01, 2.09686080e-02,
            7.07106781e-01, 1.48979724e-14, 7.07106781e-01, - 1.48999170e-14,
            # puck goal
            0, 0.6, 0.02,
            1, 0, 1, 0,
        ]

    def log_diagnostics(self, paths, logger=logger, prefix=""):
        super().log_diagnostics(paths)

        statistics = OrderedDict()
        for stat_name in [
            'hand_to_goal_dist',
            'puck_distance',
            'touch_distance',
            'sum_distance',
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

    def sample_goal_xyxy(self):
        hand = np.random.uniform(self.HAND_GOAL_LOW, self.HAND_GOAL_HIGH)
        if self.puck_limit == 'normal':
            puck = np.random.uniform(self.PUCK_GOAL_LOW, self.PUCK_GOAL_HIGH)
        elif self.puck_limit == 'large':
            puck = np.random.uniform(
                self.PUCK_GOAL_LARGE_LOW,
                self.PUCK_GOAL_LARGE_HIGH,
            )
        return np.hstack((hand, puck))

    def set_hand_xy(self, xy):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([xy[0], xy[1], 0.02]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)

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
        qpos[9:12] = np.hstack((hand_goal.copy(), np.array([0.02])))
        qvel[9:12] = [0, 0, 0]
        qpos[16:19] = np.hstack((puck_goal.copy(), np.array([0.02])))
        qvel[16:19] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def get_nongoal_state(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        mocap_pos = self.data.get_mocap_pos('mocap').copy()
        mocap_quat = self.data.get_mocap_quat('mocap').copy()
        return qpos.copy(), qvel.copy(), mocap_pos, mocap_quat

    def set_nongoal_state(self, state):
        self.set_state(state[0], state[1])
        self.data.set_mocap_pos('mocap', state[2])
        self.data.set_mocap_quat('mocap', state[3])

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

    def set_to_goal(self, goal):
        self.set_hand_xy(goal[:2])
        self.set_puck_xy(goal[2:])

    def convert_obs_to_goals(self, obs):
        return obs

    def sample_goals(self, batch_size):
        raise NotImplementedError()
