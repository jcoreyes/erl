import numpy as np
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.envs.base import Env
from rllab.misc import logger
from railrl.envs.mujoco.mujoco_env import MujocoEnv
import mujoco_py
import imp
import ipdb

JOINT_ANGLES_HIGH = np.array([
    1.70167993,
    1.04700017,
    3.0541791,
    2.61797006,
    3.05900002,
    2.09400001,
    3.05899961,
])

JOINT_ANGLES_LOW = np.array([
    -1.70167995,
    -2.14700025,
    -3.0541801,
    -0.04995198,
    -3.05900015,
    -1.5708003,
    -3.05899989
])

JOINT_VEL_HIGH = 2*np.ones(7)
JOINT_VEL_LOW = -2*np.ones(7)

JOINT_TORQUE_HIGH = 10*np.ones(8)
JOINT_TORQUE_LOW = -10*np.ones(8)

JOINT_VALUE_HIGH = {
    'position': JOINT_ANGLES_HIGH,
    'velocity': JOINT_VEL_HIGH,
    'torque': JOINT_TORQUE_HIGH,
}
JOINT_VALUE_LOW = {
    'position': JOINT_ANGLES_LOW,
    'velocity': JOINT_VEL_LOW,
    'torque': JOINT_TORQUE_LOW,
}

class SawyerEnv(MujocoEnv):
    def __init__(self, action_mode='torque'):
        self.init_serialization(locals())

        super().__init__('sawyer.xml')

        self._action_space = Box(
            JOINT_VALUE_LOW[action_mode],
            JOINT_VALUE_HIGH[action_mode]
        )

        self._observation_space = Box(
            np.hstack((JOINT_VALUE_LOW['position'], JOINT_VALUE_LOW['velocity'])),
            np.hstack((JOINT_VALUE_HIGH['position'], JOINT_VALUE_HIGH['velocity']))
        )

        # self._observation_space = Box(
        #     JOINT_VALUE_LOW['position'],
        #     JOINT_VALUE_HIGH['position']
        # )
        self.desired = np.ones(7)
        self.reset()
        
    #needs to return the observation, reward, done, and info
    def _step(self, a):
        action = np.hstack((a, [0]))
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = -np.mean((self.desired-self._get_joint_angles())**2)
        if reward < -100:
            ipdb.set_trace()
        done = False
        info = {}
        return obs, reward, done, info

    def reset(self):
        #reset to some arbitrary neutral position
        #currently it is equal to the neutral position for the baxter:
        angles = [1.2513886965340406, 0.005148737818322147,  0.004222751482764409, 0.0012673001326177769,
                  0.7523082764083187, 0.00016802203726484777, -0.5449456802340835]
        angles = [[self.wrapper(angle)] for angle in angles]
        angles = np.concatenate((angles, [[0]]), axis=0)
        # self.model.data.__setattr__('qpos', angles)
        # self.set_state(angles, Non
        velocities = np.ones(8)
        velocities = np.array([[velocity] for velocity in velocities])
        self.set_state(angles, velocities)
        return self._get_obs()

    def _get_joint_angles(self):
        return np.array([self.wrapper(angle) for angle in np.concatenate([self.model.data.qpos]).ravel()[:7]])  

    def _get_obs(self):
        # ipdb.set_trace()
        joint_pos = self._get_joint_angles()
        joint_vel = np.concatenate([self.model.data.qpos]).ravel()[:7]
        return np.hstack((joint_pos, joint_vel))

    def viewer_setup(self):
        gofast = False
        self.viewer = mujoco_py.MjViewer(visible=True, init_width=480,
                                     init_height=480, go_fast=gofast)
        self.viewer.start()
        self.viewer.set_model(self.model)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def wrapper(self, angle):
        while angle > np.pi:
            angle -= np.pi
        while angle < -np.pi:
            angle += np.pi
        return angle
#how does this environment command actions?
# See twod_point.py
#         self.do_simulation(a, self.frame_skip)
# we want the observation space to be position and velocity
"""
qpos = joint angles/position
qvel = joint velocities

Based off of 

    <actuator>
        <motor joint="right_j0" ctrlrange="-100.0 100.0" ctrllimited="true"/>
        <motor joint="right_j1" ctrlrange="-100.0 100.0" ctrllimited="true"/>
        <motor joint="right_j2" ctrlrange="-100.0 100.0" ctrllimited="true"/>
        <motor joint="right_j3" ctrlrange="-100.0 100.0" ctrllimited="true"/>
        <motor joint="right_j4" ctrlrange="-100.0 100.0" ctrllimited="true"/>
        <motor joint="right_j5" ctrlrange="-100.0 100.0" ctrllimited="true"/>
        <motor joint="right_j6" ctrlrange="-100.0 100.0" ctrllimited="true"/>
        <motor joint="head_pan" ctrlrange="-100.0 100.0" ctrllimited="true"/>
    </actuator>
    
    
in sawyer.xml
"""
#how do we get observations? and what form do they take?
#how do we reset the model?

#need to communicate actions to simulator:
#or does self.do_simulation just handle that? hmmm

#need to figure out why simulator is moving on its own
