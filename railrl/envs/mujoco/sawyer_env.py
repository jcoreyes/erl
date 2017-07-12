import numpy as np
from rllab.spaces.box import Box
from rllab.misc import logger
from railrl.envs.mujoco.mujoco_env import MujocoEnv
import mujoco_py
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
    def __init__(
            self,
            loss,
            action_mode='torque',
            delta=10,
            torque_reg_lambda=1,
            joint_angle_experiment=True,
            end_effector_experiment=False,
        ):
        self.init_serialization(locals())
        self.end_effector_experiment = end_effector_experiment
        super().__init__('sawyer.xml')
        if loss == 'MSE':
            self.MSE = True
            self.huber = False
        elif loss == 'huber':
            self.MSE = False
            self.huber = True
        self.delta=delta
        self.torque_reg_lambda = torque_reg_lambda
        self._action_space = Box(
            JOINT_VALUE_LOW[action_mode],
            JOINT_VALUE_HIGH[action_mode]
        )

        self._observation_space = Box(
            np.hstack((JOINT_VALUE_LOW['position'], JOINT_VALUE_LOW['velocity'])),
            np.hstack((JOINT_VALUE_HIGH['position'], JOINT_VALUE_HIGH['velocity']))
        )

        self.desired = np.zeros(7)
        self.reset()

    #needs to return the observation, reward, done, and info
    def _step(self, a):
        action = np.hstack((a, [0]))
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        if self.MSE:
            reward = -np.mean((self.desired-self._get_joint_angles())**2)
        elif self.huber:
            # TODO(murtaza): rename this variable
            a = np.mean(np.abs(self.desired - self._get_joint_angles()))
            if a <= self.delta:
                reward = -1 / 2 * a ** 2
            else:
                reward = -1 * self.delta * (a - 1 / 2 * self.delta)
        else:
            reward = 0
        reward -= np.linalg.norm(a)**2 * self.torque_reg_lambda
        done = False
        info = {}
        return obs, reward, done, info

    def reset(self):
        angles = [1.2513886965340406, 0.005148737818322147,  0.004222751482764409, 0.0012673001326177769,
                  0.7523082764083187, 0.00016802203726484777, -0.5449456802340835]
        angles = [[self.wrapper(angle)] for angle in angles]
        angles = np.concatenate((angles, [[0]]), axis=0)
        velocities = np.zeros((8, 1))
        self.set_state(angles, velocities)
        return self._get_obs()

    def _get_joint_angles(self):
        return np.array([self.wrapper(angle) for angle in np.concatenate([self.model.data.qpos]).ravel()[:7]])

    def _get_obs(self):
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

    def log_diagnostics(self, paths):
        if self.end_effector_experiment:
            obsSets = [path["observations"] for path in paths]
            positions = []
            desired_positions = []
            for obsSet in obsSets:
                for observation in obsSet:
                    positions.append(observation[21:24])
                    desired_positions.append(observation[24:27])

            positions = np.array(positions)
            desired_positions = np.array(desired_positions)
            mean_distance_from_desired_ee_pose = np.mean(linalg.norm(positions - desired_positions, axis=1))
            logger.record_tabular("Mean Distance from desired end-effector position",
                                  mean_distance_from_desired_ee_pose)

            if self.safety_limited_end_effector:
                mean_distance_outside_box = np.mean(
                    [self.compute_mean_distance_outside_box(pose) for pose in positions])
                logger.record_tabular("Mean Distance Outside Box", mean_distance_outside_box)


        if self.joint_angle_experiment:
            obsSets = [path["observations"] for path in paths]
            angles = []
            desired_angles = []
            positions = []
            for obsSet in obsSets:
                for observation in obsSet:
                    angles.append(observation[:7])
                    desired_angles.append(observation[24:31])
                    positions.append(observation[21:24])

            angles = np.array(angles)
            desired_angles = np.array(desired_angles)

            mean_distance_from_desired_angle = np.mean(linalg.norm(angles - desired_angles, axis=1))
            logger.record_tabular("Mean Distance from desired angle", mean_distance_from_desired_angle)

            if self.safety_limited_end_effector:
                mean_distance_outside_box = np.mean(
                    [self.compute_mean_distance_outside_box(pose) for pose in positions if not self.is_in_box(pose)])
                # ipdb.set_trace()
                logger.record_tabular("Mean Distance Outside Box", mean_distance_outside_box)
    def terminate(self):
        self.reset()

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
