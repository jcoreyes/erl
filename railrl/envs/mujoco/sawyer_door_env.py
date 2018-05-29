import abc
from collections import OrderedDict

import mujoco_py
import numpy as np
import sys
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
from railrl.core import logger
from railrl.core.serializable import Serializable
from railrl.envs.env_utils import get_asset_full_path
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths


class SawyerDoorEnv(MultitaskEnv, MujocoEnv, Serializable, metaclass=abc.ABCMeta):
    '''
    Work in Progress
    '''
    def __init__(self, frame_skip=30):
        self.quick_init(locals())
        self.min_angle = -1.5708
        self.max_angle = 1.5708
        # this should be
        self.goal_space = Box(
            np.array([self.min_angle]),
            np.array([self.max_angle]),
        )
        goal = self.sample_goal_for_rollout()
        self.set_goal(goal)
        MujocoEnv.__init__(self, self.model_path, frame_skip=frame_skip)
        obs_space_angles = 1.5708

        self.action_space = Box(
            np.array([-1, -1]),
            np.array([1, 1]),
        )

        self.observation_space = Box(
            np.array([-1, -1, -1, -obs_space_angles]),
            np.array([1, 1, 1, obs_space_angles]),
        )


        self.reset()
        self.reset_mocap_welds()

    @property
    def model_path(self):
        return get_asset_full_path('kitchen/sawyer_door.xml')

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_door_angle()
        return np.concatenate((e, b))

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.mocap_set_action(a[:3] / 100)
        u = np.zeros((8))
        u[7] = a[3]
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()
        reward = self._compute_reward(obs,self.get_goal())
        done = False
        info = dict(angle_difference=-1*reward)
        return obs, reward, done, info

    def mocap_set_action(self, action):
        pos_delta = action[None]
        self.reset_mocap2body_xpos()
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

    def get_door_angle(self):
        return np.array([self.data.get_joint_qpos('doorjoint')])

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    def _compute_reward(self, obs, goal):
        actual_angle = self.convert_ob_to_goal(obs)
        return - np.abs(actual_angle - goal)[0]

    def reset(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.init_angles
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())
        return self._get_obs()

    @property
    def init_angles(self):
        return [
            0,
            1.02866769e+00, - 6.95207647e-01, 4.22932911e-01,
            1.76670458e+00, - 5.69637604e-01, 6.24117280e-01,
            3.53404635e+00,
            0
        ]

    ''' Multitask Functions '''
    @property
    def goal_dim(self):
        return 1

    def sample_goals(self, batch_size):
        return np.reshape(np.random.uniform(self.min_angle, self.max_angle, batch_size), (batch_size, 1))

    def convert_obs_to_goals(self, obs):
        return np.reshape(obs[:, -1], (obs.shape[0], 1))

    def compute_her_reward_np(self, ob, action, next_ob, goal, env_info=None):
        return self._compute_reward(next_ob, goal)

    def set_goal_angle(self, angle):
        self._goal_angle = angle.copy()
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[0] = angle.copy()
        qvel[0] = 0
        self.set_state(qpos, qvel)

    def sample_goal_for_rollout(self):
        return self.goal_space.sample()

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        self._goal_angle = goal

    def set_to_goal(self, goal):
        self.set_goal_angle(goal)

    def get_goal(self):
        return self._goal_angle

    def log_diagnostics(self, paths, logger=logger, prefix=""):
        super().log_diagnostics(paths)

        statistics = OrderedDict()
        for stat_name in [
            'angle_difference',
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

class SawyerDoorPushOpenEnv(SawyerDoorEnv):
    def __init__(self, **kwargs):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.min_angle = 0

class SawyerDoorPullOpenEnv(SawyerDoorEnv):
    def __init__(self, **kwargs):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.max_angle = 0


if __name__ == "__main__":
    import pygame
    from pygame.locals import QUIT, KEYDOWN

    pygame.init()

    screen = pygame.display.set_mode((400, 300))

    char_to_action = {
        'w': np.array([0, -1, 0, 0]),
        'a': np.array([1, 0, 0, 0]),
        's': np.array([0, 1, 0, 0]),
        'd': np.array([-1, 0, 0, 0]),
        'q': np.array([1, -1, 0, 0]),
        'e': np.array([-1, -1, 0, 0]),
        'z': np.array([1, 1, 0, 0]),
        'c': np.array([-1, 1, 0, 0]),
        'x': 'toggle',
        'r': 'reset',
    }

    env = SawyerDoorPushOpenEnv()
    env.reset()
    ACTION_FROM = 'controller'
    # ACTION_FROM = 'pd'
    # ACTION_FROM = 'random'
    H = 100000
    # H = 300
    # H = 50


    lock_action = False
    goal = env.sample_goal_for_rollout()
    env.set_goal(goal)
    while True:
        obs = env.reset()
        last_reward_t = 0
        returns = 0
        action = np.zeros_like(env.action_space.sample())
        for t in range(H):
            done = False
            if ACTION_FROM == 'controller':
                if not lock_action:
                    action = np.array([0, 0, 0, 0])
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
                            action = np.array([0, 0, 0, 0])
                        print("got char:", char)
                        print("action", action)
                        print("angles", env.data.qpos.copy())
            else:
                action = env.action_space.sample()
            obs, reward, _, info = env.step(action)
            env.render()
            if done:
                break
        print("new episode")

