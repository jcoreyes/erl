import abc
import mujoco_py
import numpy as np
import sys
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

from railrl.core.serializable import Serializable
from railrl.envs.env_utils import get_asset_full_path
from railrl.envs.multitask.multitask_env import MultitaskEnv


class SawyerDoorEnv(MujocoEnv, Serializable, metaclass=abc.ABCMeta):
    '''
    Work in Progress
    '''
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

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.mocap_set_action(a[:3] / 100)
        u = np.zeros((8))
        u[7] = a[3]
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()
        reward = self._compute_reward(u, obs)
        done = False
        info = dict()
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

    def _get_obs(self):
        '''
        qpos contains the angles of the following joints in order:
        door
        j0-j6 of sawyer arm
        rightclaw
        dim = 9
        '''
        return np.concatenate((self.data.qpos, self.get_endeff_pos()))

    def _compute_reward(self, action, next_obs):
        pass

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
class DoorPushOpenEnv(SawyerDoorEnv, MultitaskEnv):
    def __init__(self, **kwargs):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        super().__init__(**kwargs)
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.observation_space = Box(
            np.array([-1, -1, -1, 0]),
            np.array([1, 1, 1, self.max_angle]),
        )
        self.max_angle = 1.5708

        #this should be
        self.goal_space = Box(
            np.array([0]),
            np.array([self.max_angle]),
        )

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_door_angle() % (2*np.pi) #just in case we get negative angles
        return np.concatenate((e, b))

    @property
    def goal_dim(self) -> int:
        return 1

    def sample_goals(self, batch_size):
        return np.random.uniform(0, self.max_angle, batch_size)

    def convert_obs_to_goals(self, obs):
        return obs[:, -1]

    def _compute_reward(self, action, next_obs):
        actual_angle = self.convert_ob_to_goal(next_obs)
        return - np.abs(actual_angle - self.multitask_goal)

    def set_goal_angle(self, angle):
        self._goal_angle = angle.copy()
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[0] = angle.copy()
        qvel[0] = [0]
        self.set_state(qpos, qvel)

    def sample_goal_for_rollout(self):
        return self.sample_goals(1)[0]

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        self.set_goal_angle(goal)

    def get_goal(self):
        return self._goal_angle

    def set_to_goal(self, goal):
        self.set_goal(goal)

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

    env = SawyerDoorEnv()
    env.reset()
    import ipdb; ipdb.set_trace()
    lock_action = False
    while True:
        obs = env.reset()
        last_reward_t = 0
        returns = 0
        action = np.zeros_like(env.action_space.sample())
        for t in range(H):
            done = False
            print(env.data.qpos)
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

            env.step(action)
            env.render()
            if done:
                break
        print("new episode")

'''
TODO:
setup viewer
figure out proper goal angles 
put door in correct position 
train from state on her-td3
'''
