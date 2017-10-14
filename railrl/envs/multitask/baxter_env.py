from railrl.envs.ros.baxter_env import BaxterEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv
import numpy as np

NUM_JOINTS = 7

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

JOINT_TORQUE_HIGH = 1*np.ones(7)
JOINT_TORQUE_LOW = -1*np.ones(7)

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

right_safety_box_lows = [
    0.08615153069561253,
    -1.406381785756028,
    -0.5698485041484043
]

right_safety_box_highs = [
    1.463239572333106,
    0.3499125815982429,
    0.9771420218394952,
]

left_safety_box_lows = [
    0.3404830862298487,
    -0.3499125815982429,
    -0.5698485041484043
]

left_safety_box_highs = [
    1.1163239572333106,
    1.406381785756028,
    0.795699462010194
]

joint_names = [
    '_lower_elbow',
    '_upper_forearm',
    '_lower_forearm',
    '_wrist',
]

experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]


class MultiTaskBaxterEnv(BaxterEnv, MultitaskEnv):
    #TODO: correct sampling code to have proper ranges for each measurement
    def set_goal(self, goal):
        self.desired = goal

    @property
    def goal_dim(self):
        return self.desired.size

    def sample_goal_states(self, batch_size):
        return np.random.rand(batch_size, self.desired.size)[0]

    def sample_actions(self, batch_size):
        return np.random.uniform(-1, 1, (batch_size, 7))

    def sample_states(self, batch_size):
        return np.random.rand(batch_size, self.observation_space.flat_dim)[0]

    def convert_obs_to_goal_states(self, obs):
        return obs[:, 21:24]