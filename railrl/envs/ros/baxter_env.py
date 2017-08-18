import rospy
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import baxter_interface as bi
import numpy as np
from rllab.envs.base import Env
from rllab.misc import logger
from numpy import linalg
from robot_info.srv import *
from railrl.misc.data_processing import create_stats_ordered_dict
from collections import OrderedDict
import ipdb
from experiments.murtaza.ros.joint_space_impedance import PDController

NUM_JOINTS = 7

"""
These are just ball-parks. For more specific specs, either measure them
and/or see http://sdk.rethinkrobotics.com/wiki/Hardware_Specifications.
"""

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

#not sure what the min/max angle and pos are supposed to be
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
    # '_upper_shoulder',
    # '_lower_shoulder',
    # '_upper_elbow',
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

        if loss == 'MSE':
            self.reward_function = self._MSE_reward
        elif loss == 'huber':
            self.reward_function = self._Huber_reward

        self.huber_delta = huber_delta
        self.safety_force_magnitude = safety_force_magnitude
        self.temp = temp

        self.arm = bi.Limb(self.arm_name)
        self.arm_joint_names = self.arm.joint_names()
        self.gripper = bi.Gripper(self.arm_name)

        self.PDController = PDController(robot="baxter", limb_name=self.arm_name)

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

        #set up lows and highs for observation space based on which experiment we are running
        #additionally set up the desired angle as well
        if self.joint_angle_experiment:
            lows = np.hstack((
                JOINT_VALUE_LOW['position'],
                JOINT_VALUE_LOW['velocity'],
                JOINT_VALUE_LOW['torque'],
                END_EFFECTOR_VALUE_LOW['position'],
                JOINT_VALUE_LOW['position'],
            ))

            highs = np.hstack((
                JOINT_VALUE_HIGH['position'],
                JOINT_VALUE_HIGH['velocity'],
                JOINT_VALUE_HIGH['torque'],
                END_EFFECTOR_VALUE_HIGH['position'],
                JOINT_VALUE_HIGH['position'],
            ))

            if self.fixed_angle:
                self.desired = np.zeros(NUM_JOINTS)
                # angles = {'right_s0': -0.0049903943346150115, 'right_e0': 0.007773590752427673, 'right_w2': 0.0083526057004919, 'right_w0': 0.008741701187912732, 'right_s1': -0.5451848158718935, 'right_w1': 1.2560255299115681, 'right_e1': 0.7475769104563383}
                # # angles = [0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0]
                # angles = np.array(
                #     [angles['right_s0'], angles['right_s1'], angles['right_e0'], angles['right_e1'], angles['right_w0'],
                #      angles['right_w1'], angles['right_w2']])
            else:
                self._randomize_desired_angles()

        elif self.end_effector_experiment_position:
            lows = np.hstack((
                JOINT_VALUE_LOW['position'],
                JOINT_VALUE_LOW['velocity'],
                JOINT_VALUE_LOW['torque'],
                END_EFFECTOR_VALUE_LOW['position'],
                END_EFFECTOR_VALUE_LOW['position'],
            ))

            highs = np.hstack((
                JOINT_VALUE_HIGH['position'],
                JOINT_VALUE_HIGH['velocity'],
                JOINT_VALUE_HIGH['torque'],
                END_EFFECTOR_VALUE_HIGH['position'],
                END_EFFECTOR_VALUE_HIGH['position'],
            ))

            if self.fixed_end_effector:
                self.desired = np.array([
                    1.1349147779210946,
                    -0.7649915111535125,
                    0.5545338667382815
                ])

            else:
                self._randomize_desired_end_effector_pose()

        elif self.end_effector_experiment_total:
            lows = np.hstack((
                JOINT_VALUE_LOW['position'],
                JOINT_VALUE_LOW['velocity'],
                JOINT_VALUE_LOW['torque'],
                END_EFFECTOR_VALUE_LOW['position'],
                END_EFFECTOR_VALUE_LOW['angle'],
                END_EFFECTOR_VALUE_LOW['position'],
                END_EFFECTOR_VALUE_LOW['angle'],
            ))

            highs = np.hstack((
                JOINT_VALUE_HIGH['position'],
                JOINT_VALUE_HIGH['velocity'],
                JOINT_VALUE_HIGH['torque'],
                END_EFFECTOR_VALUE_HIGH['position'],
                END_EFFECTOR_VALUE_HIGH['angle'],
                END_EFFECTOR_VALUE_HIGH['position'],
                END_EFFECTOR_VALUE_HIGH['angle'],
            ))

            if self.fixed_end_effector:
                self.desired = np.array([
                    0.14854234521312332,
                    -0.43227588084273644,
                    -0.7116727296474704,
                    0.46800638382243764,
                    0.8837008622125236,
                    -0.005641180126528841,
                    -0.0033148021158782666
                ])
            else:
                self._randomize_desired_end_effector_pose()

        self._observation_space = Box(lows, highs)

    @safe
    def _act(self, action):
        if self.safety_box:
            joint_dict = self.getRobotPoseAndJacobian()
            self.check_joints_in_box(joint_dict)
            if len(joint_dict) > 0:
                forces_dict = self.get_adjustment_forces_per_joint_dict(joint_dict)
                torques = np.zeros(7)
                for joint in forces_dict:
                    torques = torques + np.dot(joint_dict[joint][1].T, forces_dict[joint]).T
                if self.remove_action:
                    action = torques
                else:
                    action = action + torques

        joint_to_values = dict(zip(self.arm_joint_names, action))
        self._set_joint_values(joint_to_values)
        self.rate.sleep()

    def _wrap_angles(self, angles):
        return angles % (2*np.pi)

    def _joint_angles(self):
        joint_to_angles = self.arm.joint_angles()
        angles =  np.array([
            joint_to_angles[joint] for joint in self.arm_joint_names
        ])
        angles = self._wrap_angles(angles)
        return angles

    def _end_effector_pose(self):
        state_dict = self.arm.endpoint_pose()
        pos = state_dict['position']
        if self.end_effector_experiment_total:
            orientation = state_dict['orientation']
            return np.array([
                pos.x,
                pos.y,
                pos.z,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w
            ])
        else:
            return np.array([
                pos.x,
                pos.y,
                pos.z
            ])

    def _MSE_reward(self, differences):
        reward = -np.mean(differences**2)
        return reward

    def _Huber_reward(self, differences):
        a = np.mean(differences)
        if a <= self.huber_delta:
            reward = -1 / 2 * a ** 2
        else:
            reward = -1 * self.huber_delta * (a - 1 / 2 * self.huber_delta)
        return reward

    def compute_angle_difference(self, angles1, angles2):
        """
          :param angle1: A wrapped angle
          :param angle2: A wrapped angle
          """
        self._wrap_angles(angles1)
        self._wrap_angles(angles2)
        deltas = np.abs(angles1 - angles2)
        differences = np.array([min(2*np.pi-delta, delta) for delta in deltas])
        return differences

    def step(self, action):
        """
        :param huber_deltas: a change joint angles
        """
        import math
        for act in action:
            if math.isnan(act):
                print(act)
                action = np.zeros(7)
        self._act(action)
        observation = self._get_observation()
        if self.joint_angle_experiment:
            current = self._joint_angles()
            differences = self.compute_angle_difference(current, self.desired)
        elif self.end_effector_experiment_position or self.end_effector_experiment_total:
            current = self._end_effector_pose()
            differences = np.abs(current - self.desired)
        reward_function = self.reward_function
        reward = reward_function(differences)
        if self.include_torque_penalty:
            self.penalty_lambda = 1
            reward -= self.penalty_lambda * np.linalg.norm(action)
        done = False
        info = {}
        return observation, reward, done, info

    def _get_observation(self):
        angles = self._get_joint_values['angle']()
        velocities_dict = self._get_joint_values['velocity']()
        torques_dict = self._get_joint_values['torque']()
        velocities = np.array([velocities_dict[joint] for joint in self.arm_joint_names])
        torques = np.array([torques_dict[joint] for joint in self.arm_joint_names])
        temp = np.hstack((angles, velocities, torques))
        temp = np.hstack((temp, self._end_effector_pose()))
        temp = np.hstack((temp, self.desired))
        return temp

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        if self.joint_angle_experiment and not self.fixed_angle:
            self._randomize_desired_angles()
        elif self.end_effector_experiment_position \
                or self.end_effector_experiment_total and not self.fixed_end_effector:
            self._randomize_desired_end_effector_pose()

        self.arm.move_to_neutral()
        return self._get_observation()

    def _randomize_desired_angles(self):
        self.desired = np.random.rand(1, 7)[0] * 2 - 1

    def _randomize_desired_end_effector_pose(self):
        if self.end_effector_experiment_position:
            self.desired = np.random.rand(1, 3)[0] * 2 - 1
        else:
            self.desired = np.random.rand(1, 7)[0] * 2 - 1


    def _get_robot_pose_jacobian_client(self, name, tip):
        rospy.wait_for_service('get_robot_pose_jacobian')
        try:
            get_robot_pose_jacobian = rospy.ServiceProxy('get_robot_pose_jacobian', getRobotPoseAndJacobian,
                                                         persistent=True)
            resp = get_robot_pose_jacobian(name, tip)
            jacobian = np.array([
                    resp.jacobianr1,
                    resp.jacobianr2,
                    resp.jacobianr3
                    ])
            return resp.pose, jacobian
        except rospy.ServiceException as e:
            print(e)

    def getRobotPoseAndJacobian(self):
        dictionary = {}
        for joint in joint_names:
            pose, jacobian = self._get_robot_pose_jacobian_client(self.arm_name, joint)
            dictionary[self.arm_name+joint] = [pose, jacobian]
        return dictionary

    def check_joints_in_box(self, joint_dict):
        keys_to_remove = []
        for joint in joint_dict.keys():
            if self.is_in_box(joint_dict[joint][0]):
               keys_to_remove.append(joint)

        for key in keys_to_remove:
            joint_dict.pop(key)

        return joint_dict

    def is_in_box(self, endpoint_pose):
        if self.arm_name == 'right':
            within_box = [curr_pose > lower_pose and curr_pose < higher_pose
                for curr_pose, lower_pose, higher_pose
                in zip(endpoint_pose, right_safety_box_lows, right_safety_box_highs)]
        else:
            within_box = [curr_pose > lower_pose and curr_pose < higher_pose
                for curr_pose, lower_pose, higher_pose
                in zip(endpoint_pose, left_safety_box_lows, left_safety_box_highs)]
        return all(within_box)

    def get_adjustment_forces_per_joint_dict(self, joint_dict):
        forces_dict = {}
        for joint in joint_dict:
            force = self.get_adjustment_force(joint_dict[joint][0])
            forces_dict[joint] = force
        return forces_dict

    def get_adjustment_force(self, endpoint_pose):
        x, y, z = 0, 0, 0

        curr_x = endpoint_pose[0]
        curr_y = endpoint_pose[1]
        curr_z = endpoint_pose[2]
        if self.arm_name == 'right':
            if curr_x > right_safety_box_highs[0]:
                x = -1 * np.exp(np.abs(curr_x - right_safety_box_highs[0]) * self.temp) * self.safety_force_magnitude
            elif curr_x < right_safety_box_lows[0]:
                x = np.exp(np.abs(curr_x - right_safety_box_lows[0]) * self.temp) * self.safety_force_magnitude

            if curr_y > right_safety_box_highs[1]:
                y = -1 * np.exp(np.abs(curr_y - right_safety_box_highs[1]) * self.temp) * self.safety_force_magnitude
            elif curr_y < right_safety_box_lows[1]:
                y = np.exp(np.abs(curr_y - right_safety_box_lows[1]) * self.temp) * self.safety_force_magnitude

            if curr_z > right_safety_box_highs[2]:
                z = -1 * np.exp(np.abs(curr_z - right_safety_box_highs[2]) * self.temp) * self.safety_force_magnitude
            elif curr_z < right_safety_box_lows[2]:
                z = np.exp(np.abs(curr_z - right_safety_box_highs[2]) * self.temp) * self.safety_force_magnitude
        else:
            if curr_x > left_safety_box_highs[0]:
                x = -1 * np.exp(np.abs(curr_x - left_safety_box_highs[0]) * self.temp) * self.safety_force_magnitude
            elif curr_x < left_safety_box_lows[0]:
                x = np.exp(np.abs(curr_x - left_safety_box_lows[0]) * self.temp) * self.safety_force_magnitude

            if curr_y > left_safety_box_highs[1]:
                y = -1 * np.exp(np.abs(curr_y - left_safety_box_highs[1]) * self.temp) * self.safety_force_magnitude
            elif curr_y < left_safety_box_lows[1]:
                y = np.exp(np.abs(curr_y - left_safety_box_lows[1]) * self.temp) * self.safety_force_magnitude

            if curr_z > left_safety_box_highs[2]:
                z = -1 * np.exp(np.abs(curr_z - left_safety_box_highs[2]) * self.temp) * self.safety_force_magnitude
            elif curr_z < left_safety_box_lows[2]:
                z = np.exp(np.abs(curr_z - left_safety_box_highs[2]) * self.temp) * self.safety_force_magnitude


        return np.array([x, y, z])

    def compute_distances_outside_box(self, pose):
        curr_x = pose[0]
        curr_y = pose[1]
        curr_z = pose[2]
        if(self.is_in_box(pose)):
            x, y, z = 0, 0, 0
        else:
            x, y, z = 0, 0, 0
            if self.arm_name == 'right':
                if curr_x > right_safety_box_highs[0]:
                    x = np.abs(curr_x - right_safety_box_highs[0])
                elif curr_x < right_safety_box_lows[0]:
                    x = np.abs(curr_x - right_safety_box_lows[0])

                if curr_y > right_safety_box_highs[1]:
                    y = np.abs(curr_y - right_safety_box_highs[1])
                elif curr_y < right_safety_box_lows[1]:
                    y = np.abs(curr_y - right_safety_box_lows[1])

                if curr_z > right_safety_box_highs[2]:
                    z = np.abs(curr_z - right_safety_box_highs[2])
                elif curr_z < right_safety_box_lows[2]:
                    z = np.abs(curr_z - right_safety_box_lows[2])
            else:
                if curr_x > right_safety_box_highs[0]:
                    x = np.abs(curr_x - left_safety_box_highs[0])
                elif curr_x < left_safety_box_lows[0]:
                    x = np.abs(curr_x - left_safety_box_lows[0])

                if curr_y > left_safety_box_highs[1]:
                    y = np.abs(curr_y - left_safety_box_highs[1])
                elif curr_y < left_safety_box_lows[1]:
                    y = np.abs(curr_y - left_safety_box_lows[1])

                if curr_z > left_safety_box_highs[2]:
                    z = np.abs(curr_z - left_safety_box_highs[2])
                elif curr_z < left_safety_box_lows[2]:
                    z = np.abs(curr_z - left_safety_box_lows[2])
        return np.linalg.norm([x, y, z])

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def render(self):
        pass

    def log_diagnostics(self, paths):
        statistics = OrderedDict()
        stat_prefix = 'Test'
        if self.end_effector_experiment_total or self.end_effector_experiment_position:
            obsSets = [path["observations"] for path in paths]
            positions = []
            desired_positions = []
            if self.end_effector_experiment_total:
                orientations = []
                desired_orientations = []
            for obsSet in obsSets:
                for observation in obsSet:
                    positions.append(observation[21:24])
                    desired_positions.append(observation[24:27])

                    if self.end_effector_experiment_total:
                        orientations.append(observation[24:28])
                        desired_orientations.append(observation[28:])

            positions = np.array(positions)
            desired_positions = np.array(desired_positions)
            position_distances = linalg.norm(positions - desired_positions, axis=1)
            statistics.update(self._statistics_from_observations(
                position_distances,
                stat_prefix,
                'Distance from Desired End Effector Position'
            ))

            if self.safety_box:
                distances_outside_box = np.array([self.compute_distances_outside_box(pose) for pose in positions])
                statistics.update(self._statistics_from_observations(
                    distances_outside_box,
                    stat_prefix,
                    'End Effector Distance Outside Box'
                ))

            if self.end_effector_experiment_total:
                orientations_distance = linalg.norm(orientations-desired_orientations, axis=1)
                statistics.update(self._statistics_from_observations(
                    orientations_distance,
                    stat_prefix,
                    'Distance from Desired End Effector Orientation'
                ))

        if self.joint_angle_experiment:
            angle_differences, distances_outside_box = self._joint_angle_exp_info(paths)
            statistics.update(self._statistics_from_observations(
                angle_differences,
                stat_prefix,
                'Difference from Desired Joint Angle'
            ))

            if self.safety_box:
                statistics.update(self._statistics_from_observations(
                    distances_outside_box,
                    stat_prefix,
                    'End Effector Distance Outside Box'
                ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def _joint_angle_exp_info(self, paths):
        obsSets = [path["observations"] for path in paths]
        if self.joint_angle_experiment:
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

            differences = np.array([self.compute_angle_difference(angle_obs, desired_angle_obs) for angle_obs, desired_angle_obs in zip(angles, desired_angles)])
            angle_differences = np.mean(differences, axis=1)
            distances_outside_box = np.array([self.compute_distances_outside_box(pose) for pose in positions])
            return [angle_differences, distances_outside_box]

    def _statistics_from_observations(self, observation, stat_prefix, log_title):
        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            '{} {}'.format(stat_prefix, log_title),
            observation,
        ))

        return statistics

    @property
    def horizon(self):
        raise NotImplementedError

    def terminate(self):
        self.reset()

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass
