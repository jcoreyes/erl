import math
import time
from collections import OrderedDict
import numpy as np
import rospy
from numpy import linalg
from experiments.murtaza.ros.Sawyer.pd_controller import PDController
from railrl.envs.ros.sawyer_env_base import SawyerEnv
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.core.serializable import Serializable
from railrl.core import logger
from rllab.spaces.box import Box
from sawyer_control.srv import observation
from sawyer_control.msg import actions
from sawyer_control.srv import getRobotPoseAndJacobian
from rllab.envs.base import Env

JOINT_ANGLES_HIGH = np.array([
    1.70167993,
    1.04700017,
    3.0541791,
    2.61797006,
    3.05900002,
    2.09400001,
    3.05899961
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

MAX_TORQUES = 0.5 * np.array([8, 7, 6, 5, 4, 3, 2])
JOINT_TORQUE_HIGH = MAX_TORQUES
JOINT_TORQUE_LOW = -1*MAX_TORQUES

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

END_EFFECTOR_POS_LOW = [
    0.3404830862298487,
    -1.2633121086809487,
    -0.5698485041484043
]

END_EFFECTOR_POS_HIGH = [
    1.1163239572333106,
    0.003933425621414761,
    0.795699462010194
]

END_EFFECTOR_ANGLE_LOW = -1*np.ones(4)
END_EFFECTOR_ANGLE_HIGH = np.ones(4)

END_EFFECTOR_VALUE_LOW = {
    'position': END_EFFECTOR_POS_LOW,
    'angle': END_EFFECTOR_ANGLE_LOW,
}

END_EFFECTOR_VALUE_HIGH = {
    'position': END_EFFECTOR_POS_HIGH,
    'angle': END_EFFECTOR_ANGLE_HIGH,
}
class SawyerJointSpaceReachingEnv(SawyerEnv):
    def __init__(self,
                 experiment,
                 randomize_goal_on_reset=False,
                 *kwargs
                 ):
        self.randomize_goal_on_reset = randomize_goal_on_reset

    def reward(self, action):
        current = self._joint_angles()
        differences = self.compute_angle_difference(current, self.desired)
        reward = self.reward_function(differences)
        return reward

    def log_diagnostics(self, paths):
        raise NotImplementedError
    def _set_observation_space(self):
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.in_reset = True
        self.previous_angles = self._joint_angles()
        self._safe_move_to_neutral()
        self.previous_angles = self._joint_angles()
        self.in_reset = False
        if self.randomize_goal_on_reset:
            self._randomize_desired_angles()
        return self._get_observation()