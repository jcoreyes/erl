from collections import OrderedDict

from numpy.linalg import linalg

from railrl.envs.ros.baxter_env import BaxterEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv
import numpy as np
import baxter_interface as bi
import rospy

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.spaces import Box

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

RIGHT_END_EFFECTOR_POS_LOW = [
    0.3404830862298487,
    -1.2633121086809487,
    -0.5698485041484043
]

RIGHT_END_EFFECTOR_POS_HIGH = [
    1.4042843059147565,
    0.003933425621414761,
    0.795699462010194
]

LEFT_END_EFFECTOR_POS_LOW = [
    0.3404830862298487,
    -1.2633121086809487,
    -0.5698485041484043
]

LEFT_END_EFFECTOR_POS_HIGH = [
    1.360514343667115,
    0.4383921665010369,
    0.795699462010194
]

END_EFFECTOR_VALUE_LOW = {
    'right': {
        'position': RIGHT_END_EFFECTOR_POS_LOW,
        'angle': END_EFFECTOR_ANGLE_LOW,
        },
    'left': {
        'position': LEFT_END_EFFECTOR_POS_LOW,
        'angle': END_EFFECTOR_ANGLE_LOW,
    }
}

END_EFFECTOR_VALUE_HIGH = {
    'right': {
        'position': RIGHT_END_EFFECTOR_POS_HIGH,
        'angle': END_EFFECTOR_ANGLE_HIGH,
        },
    'left': {
        'position': LEFT_END_EFFECTOR_POS_HIGH,
        'angle': END_EFFECTOR_ANGLE_HIGH,
    }
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
    def __init__(
            self,
            arm_name,
            experiment,
            update_hz=20,
            action_mode='torque',
            remove_action=False,
            safety_box=False,
            loss='huber',
            huber_delta=10,
            safety_force_magnitude=2,
            temp=1.05,
            gpu=True,
            safe_reset_length=30,
            include_torque_penalty=False,
            reward_magnitude=1,
    ):
        Serializable.quick_init(self, locals())
        rospy.init_node('baxter_env', anonymous=True)
        self.rate = rospy.Rate(update_hz)
        self.terminate_experiment = False
        #defaults:
        self.joint_angle_experiment = False
        self.fixed_angle = False
        self.end_effector_experiment_position = False
        self.end_effector_experiment_total = False
        self.fixed_end_effector = False

        if experiment == experiments[0]:
            self.joint_angle_experiment=True
            self.fixed_angle = True
        elif experiment == experiments[1]:
            self.joint_angle_experiment=True
        elif experiment == experiments[2]:
            self.end_effector_experiment_position=True
            self.fixed_end_effector = True
        elif experiment == experiments[3]:
            self.end_effector_experiment_position=True
        elif experiment == experiments[4]:
            self.end_effector_experiment_total=True
            self.fixed_end_effector = True
        elif experiment == experiments[5]:
            self.end_effector_experiment_total = True

        self.safety_box = safety_box
        self.remove_action = remove_action
        self.arm_name = arm_name
        self.gpu = gpu
        self.safe_reset_length = safe_reset_length
        self.include_torque_penalty = include_torque_penalty
        self.reward_magnitude = reward_magnitude

        if loss == 'MSE':
            self.reward_function = self._MSE_reward
        elif loss == 'huber':
            self.reward_function = self._Huber_reward

        self.huber_delta = huber_delta
        self.safety_force_magnitude = safety_force_magnitude
        self.temp = temp

        self.arm = bi.Limb(self.arm_name)
        self.arm_joint_names = self.arm.joint_names()


        #create a dictionary whose values are functions that set the appropriate values
        action_mode_dict = {
            'position': self.arm.set_joint_positions,
            'velocity': self.arm.set_joint_velocities,
            'torque': self.arm.set_joint_torques,
        }

        #create a dictionary whose values are functions that return the appropriate values
        observation_mode_dict = {
            'angle': self._joint_angles,
            'velocity': self.arm.joint_velocities,
            'torque': self.arm.joint_efforts,
        }

        self._set_joint_values = action_mode_dict[action_mode]
        self._get_joint_values = observation_mode_dict

        self._action_space = Box(
            JOINT_VALUE_LOW[action_mode],
            JOINT_VALUE_HIGH[action_mode]
            )

        lows = np.hstack((
            JOINT_VALUE_LOW['position'],
            JOINT_VALUE_LOW['velocity'],
            JOINT_VALUE_LOW['torque'],
            END_EFFECTOR_VALUE_LOW[self.arm_name]['position'],
        ))

        highs = np.hstack((
            JOINT_VALUE_HIGH['position'],
            JOINT_VALUE_HIGH['velocity'],
            JOINT_VALUE_HIGH['torque'],
            END_EFFECTOR_VALUE_HIGH[self.arm_name]['position'],
        ))

        self._observation_space = Box(lows, highs)
    def set_goal(self, goal):
        self.desired = goal

    @property
    def goal_dim(self):
        return 3

    def sample_goal_states(self, batch_size):
        return np.random.uniform(END_EFFECTOR_POS_LOW, END_EFFECTOR_POS_HIGH, size=(batch_size, 3))

    def sample_actions(self, batch_size):
        return np.random.uniform(JOINT_VALUE_LOW['torque'], JOINT_VALUE_HIGH['torque'], (batch_size, 7))

    def sample_states(self, batch_size):
        return np.hstack((np.zeros(batch_size, 21), self.sample_goal_states(batch_size)))

    def convert_obs_to_goal_states(self, obs):
        return obs[:, 21:24]

    def _get_observation(self):
        angles = self._get_joint_values['angle']()
        velocities_dict = self._get_joint_values['velocity']()
        torques_dict = self._get_joint_values['torque']()
        velocities = np.array([velocities_dict[joint] for joint in self.arm_joint_names])
        torques = np.array([torques_dict[joint] for joint in self.arm_joint_names])
        temp = np.hstack((angles, velocities, torques))
        temp = np.hstack((temp, self._end_effector_pose()))
        return temp

    def log_diagnostics(self, paths):
        goal_states = np.vstack([path['goal_states'] for path in paths])
        statistics = OrderedDict()
        stat_prefix = 'Test'
        obsSets = [path["observations"] for path in paths]
        positions = []
        for obsSet in obsSets:
            for observation in obsSet:
                positions.append(observation[21:24])
        positions = np.array(positions)
        desired_positions = goal_states
        position_distances = linalg.norm(positions - desired_positions, axis=1)
        statistics.update(self._statistics_from_observations(
            position_distances,
            stat_prefix,
            'Distance from Desired End Effector Position'
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

