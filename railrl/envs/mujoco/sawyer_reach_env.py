from collections import OrderedDict
import numpy as np
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import mujoco_py

from railrl.core import logger

from railrl.core.serializable import Serializable
from railrl.envs.env_utils import get_asset_full_path
from railrl.envs.multitask.multitask_env import MultitaskEnv, MultitaskToFlatEnv
from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths


class SawyerReachXYZEnv(MujocoEnv, Serializable, MultitaskEnv):
    """Implements a 3D-position controlled Sawyer environment"""

    def __init__(self, reward_info=None, frame_skip=30,
                 pos_action_scale=1. / 100, hide_goal=False):
        self.quick_init(locals())
        self.reward_info = reward_info
        self.hide_goal = hide_goal
        self._goal_xyz = self.sample_goal_xyz()
        self._pos_action_scale = pos_action_scale
        MultitaskEnv.__init__(self, distance_metric_order=2)
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)

        self.action_space = Box(
            np.array([-1, -1, -1]),
            np.array([1, 1, 1]),
        )

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
    def model_name(self):
        if self.hide_goal:
            return get_asset_full_path('sawyer_reach_mocap_goal_hidden.xml')
        else:
            return get_asset_full_path('sawyer_reach_mocap.xml')

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
        self.mocap_set_action(a[:3] * self._pos_action_scale, relative=True)
        u = np.zeros((7))
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs, u, obs, self._goal_xyz)
        done = False

        distance = np.linalg.norm(self.get_goal_pos() - self.get_endeff_pos())
        info = dict(
            distance=distance,
        )
        return obs, reward, done, info

    def _get_obs(self):
        p = self.get_endeff_pos()
        return p

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def get_goal_pos(self):
        return self.data.body_xpos[self.goal_id].copy()

    def sample_goal_xyz(self):
        pos = np.random.uniform(
            np.array([-0.1, 0.5, 0.02]),
            np.array([0.1, 0.7, 0.2]),
        )
        return pos

    def set_goal_xyz(self, pos):
        self._goal_xyz = pos.copy()
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[7:10] = pos.copy()
        qvel[7:10] = [0, 0, 0]
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

    def mocap_set_action(self, action, relative=True):
        pos_delta = action[None]

        if relative:
            self.reset_mocap2body_xpos()
            new_mocap_pos = self.data.mocap_pos + pos_delta
        else:
            new_mocap_pos = pos_delta
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

    def reset(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.init_angles
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())
        self.set_goal_xyz(self._goal_xyz)
        return self._get_obs()

    def compute_reward(self, ob, action, next_ob, goal):
        if not self.reward_info or self.reward_info["type"] == "euclidean":
            r = -np.linalg.norm(next_ob - goal)
        elif self.reward_info["type"] == "sparse":
            t = self.reward_info["threshold"]
            r = float(np.linalg.norm(next_ob - goal) < t)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def compute_her_reward_np(self, ob, action, next_ob, goal, env_info=None):
        return self.compute_reward(ob, action, next_ob, goal)

    @property
    def init_angles(self):
        return [
            1.02866769e+00, - 6.95207647e-01, 4.22932911e-01,
            1.76670458e+00, - 5.69637604e-01, 6.24117280e-01,
            3.53404635e+00,
            1.07586388e-02, 6.62018003e-01, 2.09936716e-02,
            1.00000000e+00, 3.76632959e-14, 1.36837913e-11, 1.56567415e-23
        ]

    @property
    def goal_dim(self):
        return 3

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    @property
    def goal_id(self):
        return self.model.body_names.index('goal')

    def log_diagnostics(self, paths, logger=logger, prefix=""):
        super().log_diagnostics(paths)

        statistics = OrderedDict()
        for stat_name in [
            'distance',
        ]:
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
        return 3

    def sample_goal_for_rollout(self):
        return self.sample_goal_xyz()

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        self.set_goal_xyz(goal)
        # self.set_to_goal(goal)

    def get_goal(self):
        return self._goal_xyz

    def convert_obs_to_goals(self, obs):
        return obs

    def sample_goals(self, batch_size):
        raise NotImplementedError()

    def set_to_goal(self, goal):
        self._set_hand_xyz(goal)

    def _set_hand_xyz(self, xyz):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array(xyz))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)


class SawyerReachXYEnv(SawyerReachXYZEnv):
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

    def sample_goal_xyz(self):
        pos = np.random.uniform(
            np.array([-0.1, 0.5, 0.02]),
            np.array([0.1, 0.7, 0.02]),
        )
        return pos

    def step(self, a):
        a = np.clip(a, -1, 1)
        mocap_delta_z = 0.06 - self.data.mocap_pos[0, 2]
        new_mocap_action = np.hstack((
            a,
            np.array([mocap_delta_z])
        ))
        return super().step(new_mocap_action)

    def mocap_set_action(self, action, relative=True):
        pos_delta = action[None]

        if relative:
            self.reset_mocap2body_xpos()
            new_mocap_pos = self.data.mocap_pos + pos_delta
        else:
            new_mocap_pos = pos_delta
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
            0.06,
            0,
            0.5,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))


if __name__ == "__main__":
    import pygame
    from pygame.locals import QUIT, KEYDOWN

    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    char_to_action = {
        'w': np.array([0 , -1, 0 , 0]),
        'a': np.array([1 , 0 , 0 , 0]),
        's': np.array([0 , 1 , 0 , 0]),
        'd': np.array([-1, 0 , 0 , 0]),
        'q': np.array([1 , -1 , 0 , 0]),
        'e': np.array([-1 , -1 , 0, 0]),
        'z': np.array([1 , 1 , 0 , 0]),
        'c': np.array([-1 , 1 , 0 , 0]),
        # 'm': np.array([1 , 1 , 0 , 0]),
        'j': np.array([0 , 0 , 1 , 0]),
        'k': np.array([0 , 0 , -1 , 0]),
        'x': 'toggle',
        'r': 'reset',
    }

    ACTION_FROM = 'controller'
    H = 100000
    # ACTION_FROM = 'random'
    # H = 300
    # ACTION_FROM = 'pd'
    # H = 50

    env = SawyerReachXYZEnv()
    # env = SawyerReachXYEnv()
    env = MultitaskToFlatEnv(env)

    lock_action = False
    while True:
        obs = env.reset()
        last_reward_t = 0
        returns = 0
        action = np.zeros_like(env.action_space.sample())
        for t in range(H):
            done = False
            if ACTION_FROM == 'controller':
                if not lock_action:
                    action = np.array([0,0,0,0])
                for event in pygame.event.get():
                    event_happened = True
                    if event.type == QUIT:
                        sys.exit()
                    if event.type == KEYDOWN:
                        char = event.dict['key']
                        new_action = char_to_action.get(chr(char), None)
                        if new_action == 'toggle':
                            lock_action = not lock_action
                        elif new_action == 'reset':
                            done = True
                        elif new_action is not None:
                            action = new_action
                        else:
                            action = np.array([0 , 0 , 0 , 0])
                        print("got char:", char)
                        print("action", action)
                        print("angles", env.data.qpos.copy())
            elif ACTION_FROM == 'random':
                action = env.action_space.sample()
            else:
                delta = (env.get_block_pos() - env.get_endeff_pos())[:2]
                action[:2] = delta * 100
            # if t == 0:
            #     print("goal is", env.get_goal_pos())
            obs, reward, _, info = env.step(action)

            env.render()
            if done:
                break
        print("new episode")
