import rospy
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import baxter_interface as bi
import numpy as np
from cached_property import cached_property
from rllab.envs.base import Env

NUM_JOINTS = 7

"""
These are just ball-parks. For more specific specs, either measure them
and/or see http://sdk.rethinkrobotics.com/wiki/Hardware_Specifications.
"""
JOINT_ANGLES_HIGH = np.array(
    [1.70167993, 1.04700017, 3.0541791, 2.61797006, 3.05900002,
     2.09400001, 3.05899961])
JOINT_ANGLES_LOW = np.array([
    -1.70167995, -2.14700025, -3.0541801, -0.04995198, -3.05900015,
    -1.5708003, -3.05899989
])

JOINT_VEL_HIGH = 2*np.ones(7)
JOINT_VEL_LOW = -2*np.ones(7)

JOINT_TORQUE_HIGH = 10*np.ones(7)
JOINT_TORQUE_LOW = -10*np.ones(7)

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


def safe(raw_function):
    def safe_function(*args, **kwargs):
        try:
            return raw_function(*args, **kwargs)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    return safe_function


class BaxterEnv(Env, Serializable):
    def __init__(
            self,
            update_hz=20,
            action_mode='torque',
            observation_mode='position',
    ):
        Serializable.quick_init(self, locals())
        rospy.init_node('baxter_env', anonymous=True)
        self.right_arm = bi.Limb('right')
        self.right_joint_names = self.right_arm.joint_names()
        self.right_grip = bi.Gripper('right', bi.CHECK_VERSION)
        self.rate = rospy.Rate(update_hz)
        action_mode_dict = {
            'position': self.right_arm.set_joint_positions,
            'velocity': self.right_arm.set_joint_velocities,
            'torque': self.right_arm.set_joint_torques,
        }
        observation_mode_dict = {
            'position': self.right_arm.joint_angles,
            'velocity': self.right_arm.joint_velocities,
            'torque': self.right_arm.joint_efforts,
        }
        self._set_joint_values = action_mode_dict[action_mode]
        self._get_joint_to_value_dict = observation_mode_dict[observation_mode]
        self._action_space = Box(
            JOINT_VALUE_LOW[action_mode],
            JOINT_VALUE_HIGH[action_mode],
        )
        self._observation_space = Box(
            JOINT_VALUE_LOW[observation_mode],
            JOINT_VALUE_HIGH[observation_mode],
        )
        self.desired_angles = np.zeros(NUM_JOINTS)

    @safe
    def _act(self, action):
        joint_to_values = dict(zip(self.right_joint_names, action))
        self._set_joint_values(joint_to_values)
        self.rate.sleep()

    def _joint_angles(self):
        joint_to_angles = self.right_arm.joint_angles()
        return np.array([
            joint_to_angles[joint] for joint in self.right_joint_names
        ])

    def step(self, action):
        """
        :param deltas: a change joint angles
        """
        self._act(action)
        observation = self._get_joint_values()

        reward = -np.mean((self._joint_angles() - self.desired_angles)**2)
        done = False
        info = {}
        return observation, reward, done, info

    def _get_joint_values(self):
        joint_values_dict = self._get_joint_to_value_dict()
        return np.array([
            joint_values_dict[joint] for joint in self.right_joint_names
        ])

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.right_arm.move_to_neutral()
        return self._get_joint_values()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def render(self):
        pass

    def log_diagnostics(self, paths):
        pass

    @property
    def horizon(self):
        raise NotImplementedError

    def terminate(self):
        self.reset()

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass
