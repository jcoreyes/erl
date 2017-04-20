import rospy
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import baxter_interface as bi
import numpy as np
from cached_property import cached_property
from rllab.envs.base import Env

NUM_JOINTS = 7
JOINT_ANGLES_HIGH = np.array(
    [1.70167993, 1.04700017, 3.0541791, 2.61797006, 3.05900002,
     2.09400001, 3.05899961]) / 10
JOINT_ANGLES_LOW = np.array([
    -1.70167995, -2.14700025, -3.0541801, -0.04995198, -3.05900015,
    -1.5708003, -3.05899989
]) / 10


def safe(raw_function):
    def safe_function(*args, **kwargs):
        try:
            return raw_function(*args, **kwargs)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    return safe_function


class BaxterEnv(Env, Serializable):
    def __init__(self, update_hz=20):
        Serializable.quick_init(self, locals())
        rospy.init_node('baxter_env', anonymous=True)
        # self.left_arm = bi.Limb('left')
        # self.left_joint_names = self.left_arm.joint_names()
        # self.left_grip = bi.Gripper('left', bi.CHECK_VERSION)
        self.right_arm = bi.Limb('right')
        self.right_joint_names = self.right_arm.joint_names()
        self.right_grip = bi.Gripper('right', bi.CHECK_VERSION)
        self.rate = rospy.Rate(update_hz)

    @safe
    def change_joint(self, deltas):
        current_positions = self.right_arm.joint_angles()
        joint_command = {}
        joint_to_delta = dict(zip(self.right_joint_names, deltas))
        for joint_name, current_pos in current_positions.items():
            joint_command[joint_name] = current_pos + joint_to_delta[joint_name]
        self.right_arm.set_joint_positions(joint_command)

    @safe
    def set_joint_positions(self, positions):
        joint_to_positions = dict(zip(self.right_joint_names, positions))
        self.right_arm.set_joint_positions(joint_to_positions)

    @safe
    def set_joint_velocities(self, velocities):
        joint_to_velocities = dict(zip(self.right_joint_names, velocities))
        self.right_arm.set_joint_velocities(joint_to_velocities)

    @safe
    def set_joint_torques(self, torques):
        joint_to_torques = dict(zip(self.right_joint_names, torques))
        self.right_arm.set_joint_torques(joint_to_torques)

    def step(self, deltas):
        """
        :param deltas: a change joint angles
        """
        self.set_joint_positions(
            deltas,
        )
        self.rate.sleep()
        observation = self._get_joint_angles()
        reward = 0
        done = False
        info = {}
        return observation, reward, done, info

    def _get_joint_angles(self):
        joint_angles_dict = self.right_arm.joint_angles()
        return np.array([
            joint_angles_dict[joint] for joint in self.right_joint_names
        ])

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        # self.left_arm.move_to_neutral()
        self.right_arm.move_to_neutral()
        return self._get_joint_angles()

    @cached_property
    def action_space(self):
        return Box(JOINT_ANGLES_HIGH, JOINT_ANGLES_LOW)

    @cached_property
    def observation_space(self):
        return Box(JOINT_ANGLES_HIGH, JOINT_ANGLES_LOW)

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
