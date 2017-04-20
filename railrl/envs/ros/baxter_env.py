import rospy
from rllab.spaces.box import Box
from beginner_tutorials.srv import AddTwoInts, AddTwoIntsResponse
import baxter_interface as bi
import numpy as np
from cached_property import cached_property

NUM_JOINTS = 7
JOINT_ANGLES_HIGH = np.array([ 1.70167993,  1.04700017,  3.0541791 ,  2.61797006,  3.05900002,
        2.09400001,  3.05899961])
JOINT_ANGLES_LOW = np.array([
    -1.70167995, -2.14700025, -3.0541801 , -0.04995198, -3.05900015,
    -1.5708003 , -3.05899989
])

def safe(function):
    def safe_function(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
    return safe_function


@safe
def change_joint(limb, joint_names, deltas):
    current_positions = limb.joint_angles()
    joint_command = {}
    joint_to_delta = dict(zip(joint_names, deltas))
    for joint_name, current_pos in current_positions.items():
        joint_command[joint_name] = current_pos + joint_to_delta[joint_name]
    limb.set_joint_positions(joint_command)


def set_joint_positions(limb, joint_names, positions):
    joint_to_positions = dict(zip(joint_names, positions))
    limb.set_joint_positions(joint_to_positions)

class BaxterEnv(object):
    def __init__(self, update_hz=20):
        rospy.init_node('baxter_env', anonymous=True)
        # self.left_arm = bi.Limb('left')
        # self.left_joint_names = self.left_arm.joint_names()
        # self.left_grip = bi.Gripper('left', bi.CHECK_VERSION)
        self.right_arm = bi.Limb('right')
        self.right_joint_names = self.right_arm.joint_names()
        self.right_grip = bi.Gripper('right', bi.CHECK_VERSION)
        self.rate = rospy.Rate(update_hz)

    def step(self, deltas):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        change_joint(
            self.right_arm,
            self.right_joint_names,
            deltas,
        )
        self.rate.sleep()
        joint_angles_dict = self.right_arm.joint_angles()
        observation = np.array([
            joint_angles_dict[joint] for joint in self.right_joint_names
        ])
        reward = 0
        done = False
        info = {}
        return observation, reward, done, info

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        # self.left_arm.move_to_neutral()
        self.right_arm.move_to_neutral()

    @cached_property
    def action_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        return Box(JOINT_ANGLES_HIGH, JOINT_ANGLES_LOW)

    @cached_property
    def observation_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        return Box(JOINT_ANGLES_HIGH, JOINT_ANGLES_LOW)

    # Helpers that derive from Spaces
    @property
    def action_dim(self):
        return self.action_space.flat_dim

    def render(self):
        pass

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def horizon(self):
        """
        Horizon of the environment, if it has one
        """
        raise NotImplementedError


    def terminate(self):
        """
        Clean up operation,
        """
        pass

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass
