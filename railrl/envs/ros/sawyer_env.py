import rospy
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import intera_interface as ii
import numpy as np
from rllab.envs.base import Env
from rllab.misc import logger
from numpy import linalg
from robot_info.srv import *
from railrl.misc.data_processing import create_stats_ordered_dict
from collections import OrderedDict
import ipdb
from experiments.murtaza.ros.joint_space_impedance import PDController
from intera_core_msgs.msg import SEAJointState
import datetime
from gravity_torques.srv import *
import time
from queue import Queue
from intera_interface import CHECK_VERSION

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

# box_lows = [
#     0.1628008448954529,
#     -0.33786487626917794,
#     0.20084391863426093,
# ]
#
# box_highs = [
#     0.7175958839273338,
#     0.3464466563902636,
#     0.7659791453416877,
# ]

box_lows = [
    0.1328008448954529,
    -0.23786487626917794,
    0.2084391863426093,
]

box_highs = [
    0.6175958839273338,
    0.2464466563902636,
    0.8659791453416877,
]

# box_lows = [
#     0.5229248368361292,
#     -0.18738329161934256,
#     0.5247724684655974
# ]
#
# box_highs = [
#     0.7229248368361292,
#     0.2056984254930084,
#     0.7668282486663502,
# ]

joint_names = [
    # '_l0',
    # '_l1',
    '_l2',
    '_l3',
    '_l4',
    '_l5',
    '_l6',
    '_hand'
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


class SawyerEnv(Env, Serializable):
    def __init__(
            self,
            arm_name,
            experiment,
            update_hz=20,
            action_mode='torque',
            remove_action=False,
            safety_box=True,
            loss='huber',
            huber_delta=10,
            safety_force_magnitude=2,
            temp=1.05,
            safe_reset_length=50,
    ):

        Serializable.quick_init(self, locals())
        rospy.init_node('sawyer_env', anonymous=True)
        self.rate = rospy.Rate(update_hz)

        #defaults:
        self.joint_angle_experiment = False
        self.fixed_angle = False
        self.end_effector_experiment_position = False
        self.end_effector_experiment_total = False
        self.fixed_end_effector = False
        self.safety_box = False
        self.use_safety_checks = False

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
        self.safe_reset_length=safe_reset_length

        if loss == 'MSE':
            self.reward_function = self._MSE_reward
        elif loss == 'huber':
            self.reward_function = self._Huber_reward

        self.huber_delta = huber_delta
        self.safety_force_magnitude = safety_force_magnitude
        self.temp = temp

        self.arm = ii.Limb(self.arm_name)
        self.arm_joint_names = self.arm.joint_names()

        self.PDController = PDController()

        #create a dictionary whose values are functions that set the appropriate values
        action_mode_dict = {
            'angle': self.arm.set_joint_positions,
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
                JOINT_VALUE_LOW['torque'],
            ))

            highs = np.hstack((
                JOINT_VALUE_HIGH['position'],
                JOINT_VALUE_HIGH['velocity'],
                JOINT_VALUE_HIGH['torque'],
                END_EFFECTOR_VALUE_HIGH['position'],
                JOINT_VALUE_HIGH['position'],
                JOINT_VALUE_HIGH['torque'],
            ))

            if self.fixed_angle:
                angles = {
                    'right_j6': 3.23098828125,
                    'right_j5': -2.976708984375,
                    'right_j4': -0.100001953125,
                    'right_j3': 1.59925,
                    'right_j2': -1.6326630859375,
                    'right_j1': -0.3456298828125,
                    'right_j0': 0.0382529296875
                }
                angles = np.array([
                    angles['right_j0'],
                    angles['right_j1'],
                    angles['right_j2'],
                    angles['right_j3'],
                    angles['right_j4'],
                    angles['right_j5'],
                    angles['right_j6']
                ])
                angles = self._wrap_angles(angles)
                self.desired = angles
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
                des_pos = Point(x=0.44562573898386176, y=-0.055317682301721766, z=0.4950886597008108)
                self.desired = np.array([
                    des_pos.x,
                    des_pos.y,
                    des_pos.z,
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
                des = {
                    'position': Point(x=0.44562573898386176, y=-0.055317682301721766, z=0.4950886597008108),
                    'orientation': Quaternion(x=-0.5417504106748736, y=0.46162598289085305, z=0.35800013141940035, w=0.6043540769758675)}
                self.desired = np.array([
                    des['position'].x,
                    des['position'].y,
                    des['position'].z,
                    des['orientation'].x,
                    des['orientation'].y,
                    des['orientation'].z,
                    des['orientation'].w,
                ])
            else:
                self._randomize_desired_end_effector_pose()

        self._observation_space = Box(lows, highs)
        self.previous_torques = np.zeros(7)
        self.previous_positions = np.array([
            np.ones(3),
            np.ones(3),
            np.ones(3),
            np.ones(3),
            np.ones(3),
            np.ones(3),
            np.ones(3),
        ])
        self.N = 1000
        self.init_delay = time.time()
        self._rs = ii.RobotEnable(CHECK_VERSION)
        self.q = []
        self.update_pose_and_jacobian_dict()
        self.filter_actions = False
        self.filter_observations = False
        self.previous_action = np.zeros(7)
        self.previous_temp = np.zeros(38)

    @safe
    def _act(self, action):
        if self.safety_box:
            self.update_pose_and_jacobian_dict()
            self.check_joints_in_box(self.pose_jacobian_dict)
            if len(self.pose_jacobian_dict) > 0:
                forces_dict = self.get_adjustment_forces_per_joint_dict(self.pose_jacobian_dict)
                torques = np.zeros(7)
                for joint in forces_dict:
                    torques = torques + np.dot(self.pose_jacobian_dict[joint][1].T, forces_dict[joint]).T
                if self.remove_action:
                    action = torques
                else:
                    action = action + torques
        if self.filter_actions:
            action = self.filter_action(action)
        np.clip(action, -10, 10, out=action)
        joint_to_values = dict(zip(self.arm_joint_names, action))
        self._set_joint_values(joint_to_values)
        self.rate.sleep()
        return action

    def filter_action(self, action):
        return action*.9 + self.previous_action * .1

    def is_in_correct_position(self):
        desired_neutral = np.array([
            6.28115601e+00,
            5.10141089e+00,
            6.28014234e+00,
            2.17755176e+00,
            9.48242187e-04,
            5.73669922e-01,
            3.31514160e+00
        ])
        actual_neutral = self._joint_angles()
        errors = np.abs(desired_neutral - actual_neutral)
        ERROR_THRESHOLD = 1*np.ones(7)
        is_within_threshold = (errors < ERROR_THRESHOLD).all()
        return is_within_threshold

    def get_observed_torques_minus_gravity(self):
        gravity_torques, actual_effort, subtracted_torques = self._get_gravity_compensation_torques()
        return subtracted_torques

    def gravity_torques_client(self):
        rospy.wait_for_service('gravity_torques')
        try:
            gravity_torques = rospy.ServiceProxy('gravity_torques', GravityTorques)
            resp1 = gravity_torques()
            return np.array(resp1.gravity_torques), np.array(resp1.actual_torques), np.array(resp1.subtracted_torques)
        except rospy.ServiceException as e:
            print(e)

    def _get_gravity_compensation_torques(self):
        gravity_torques, actual_torques, subtracted_torques = self.gravity_torques_client()
        return gravity_torques, actual_torques, subtracted_torques

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
        if self._rs.state().stopped:
            ipdb.set_trace()
        actual_commanded_action = self._act(action)

        curr_time = time.time()
        delay = curr_time - self.init_delay
        self.init_delay = curr_time

        observation = self._get_observation()

        self.update_n_step_buffer(observation, action, delay)

        if self.joint_angle_experiment:
            current = self._joint_angles()
            differences = self.compute_angle_difference(current, self.desired)
        elif self.end_effector_experiment_position or self.end_effector_experiment_total:
            current = self._end_effector_pose()
            differences = np.abs(current - self.desired)

        reward = self.reward_function(differences)
        if self.use_safety_checks:
            out_of_box = self.safety_box_check(reset_on_error=True)
            high_torque = self.high_torque_check(actual_commanded_action, reset_on_error=True)
            # unexpected_torque = self.unexpected_torque_check(reset_on_error=True)
            unexpected_velocity = self.unexpected_velocity_check(reset_on_error=True)
            done = out_of_box  or unexpected_velocity
        else:
            done = False
        info = {}
        return observation, reward, done, info

    def safety_box_check(self, reset_on_error=True):
        self.check_joints_in_box(self.pose_jacobian_dict)
        terminate_episode = False
        # print(self.pose_jacobian_dict.values())
        if len(self.pose_jacobian_dict) > 0:
            for joint in self.pose_jacobian_dict.keys():
                dist = self.compute_distances_outside_box(self.pose_jacobian_dict[joint][0])
                if dist > .185:
                    if reset_on_error:
                        print('safety box failure during train/eval: ', joint, dist)
                        terminate_episode = True
                    else:
                        raise EnvironmentError('safety box failure during reset: ', joint, dist)
        return terminate_episode

    def jacobian_check(self):
        ee_jac = self.pose_jacobian_dict['right_hand'](1)
        if np.linalg.det(ee_jac) == 0:
            self._act(self._randomize_desired_angles())

    def update_pose_and_jacobian_dict(self):
        self.pose_jacobian_dict = self._get_robot_pose_jacobian_client('right')

    def unexpected_torque_check(self, reset_on_error=True):
        #we care about the torque that was applied to make sure it hasn't gone too high
        new_torques = self.get_observed_torques_minus_gravity()
        ERROR_THRESHOLD = np.array([5,  13, 12, 12, 10, 14, 10])
        is_peaks = (np.abs(new_torques) > ERROR_THRESHOLD).any()
        if is_peaks:
            print('unexpected_torque: ', new_torques)
            if reset_on_error:
                # print('gravity torques', self._get_gravity_compensation_torques())
                # raise EnvironmentError('unexpected torques: ', new_torques)
                return True
            else:
                # print('gravity torques', self._get_gravity_compensation_torques())
                raise EnvironmentError('unexpected torques: ', new_torques)
        return False
        #if the new_torque is significantly greater than the previous one

    def unexpected_velocity_check(self, reset_on_error=True):
        velocities_dict = self._get_joint_values['velocity']()
        velocities = np.array([velocities_dict[joint] for joint in self.arm_joint_names])
        ERROR_THRESHOLD = 5 * np.ones(7)
        is_peaks = (np.abs(velocities) > ERROR_THRESHOLD).any()
        if is_peaks:
            print('unexpected_velocities: ', velocities)
            if reset_on_error:
                # raise EnvironmentError('unexpected velocities: ', velocities)
                return True
            else:
                print('unexpected_velocities: ', velocities)
                self._rs.disable()
                self._rs.reset()
                self._rs.enable()
                # raise EnvironmentError('unexpected velocities: ', velocities)
        return False

    def get_positions_from_pose_jacobian_dict(self):
        poses = []
        for joint in self.pose_jacobian_dict.keys():
            poses.append(self.pose_jacobian_dict[joint][0])
        return np.array(poses)

    def high_torque_check(self, commanded_torques, reset_on_error=True):
        #we care about the torque that we are trying to apply
        new_torques = commanded_torques
        current_angles = self._joint_angles()
        # current_positions = self.get_positions_from_pose_jacobian_dict()
        # position_deltas = np.abs(current_positions - self.previous_positions)
        position_deltas = np.abs(current_angles - self.previous_angles) #self.previous_angles can be a moving average of the previous angles
        deltas = []
        for joint_position in position_deltas:
            deltas = deltas + joint_position
        # DELTA_THRESHOLD = .1 * np.ones(21)
        DELTA_THRESHOLD = .1 * np.ones(7)
        ERROR_THRESHOLD = np.array([10, 13, 12, 12, 10, 10, 10])
        violation = False
        for i in range(len(new_torques)):
            if new_torques[i] > ERROR_THRESHOLD[i] and position_deltas[i] < DELTA_THRESHOLD[i]:
                violation=True
        if reset_on_error and violation:
            print('high_torque:', new_torques)
            print('positions', position_deltas)
            # terminate_episode = True
            # raise EnvironmentError('ERROR: Applying large torques and not moving')
            return True
        elif violation:
            print('high_torque:', new_torques)
            print('positions', position_deltas)
            raise EnvironmentError('ERROR: Applying large torques and not moving')

        self.previous_angles = current_angles
        # self.previous_positions = .1 * self.previous_positions + .9 * current_positions
        return False

    def _get_observation(self):
        angles = self._get_joint_values['angle']()
        torques_dict = self._get_joint_values['torque']()
        velocities_dict = self._get_joint_values['velocity']()
        velocities = np.array([velocities_dict[joint] for joint in self.arm_joint_names])
        torques = np.array([torques_dict[joint] for joint in self.arm_joint_names])
        # torques = self.get_observed_torques_minus_gravity()
        temp = np.hstack((angles, velocities, torques))
        temp = np.hstack((temp, self._end_effector_pose()))
        temp = np.hstack((temp, self.desired))
        # if action is not None:
        #     temp = np.hstack((temp, action))
        gravity_torques, observed_torques, observed_torques_minus_gravity = self._get_gravity_compensation_torques()
        temp = np.hstack((temp, gravity_torques))
        if self.filter_observations:
            temp = self.previous_temp*.2 + temp * .8
        return temp

    def safe_move_to_neutral(self):
        for _ in range(self.safe_reset_length):
            torques = self.PDController._update_forces()
            actual_commanded_actions = self._act(torques)
            if self.use_safety_checks:
                self.safety_box_check(reset_on_error=False)
                # self.unexpected_torque_check(reset_on_error=False)
                self.high_torque_check(actual_commanded_actions, reset_on_error=False)
                self.unexpected_velocity_check(reset_on_error=False)

    def update_n_step_buffer(self, obs, action, delay):
        # obs_dict = {
        #     # 'angles': obs[:7],
        #     # 'velocities':obs[7:14],
        #     'observed_torques':obs[14:21],
        #     # 'ee_pose':obs[21:24],
        #     # 'desired':obs[24:31],
        #     'applied_torques':action,
        #     'delay':delay,
        # }

        gravity_torques, actual, subtracted = self._get_gravity_compensation_torques()
        obs_dict = {
            'observed':obs[14:21],
            'gravity':gravity_torques,
            'subtracted':subtracted,
            'applied':action,
        }
        self.q.append(obs_dict)
        # if len(self.q) == self.N:
        #     self.q = self.q[1:]

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.previous_angles = self._joint_angles()
        # self.previous_positions = self.get_positions_from_pose_jacobian_dict()

        if self.joint_angle_experiment and not self.fixed_angle:
            self._randomize_desired_angles()
        elif self.end_effector_experiment_position \
                or self.end_effector_experiment_total and not self.fixed_end_effector:
            self._randomize_desired_end_effector_pose()

        self.safe_move_to_neutral()
        # self.previous_positions = self.get_positions_from_pose_jacobian_dict()
        self.previous_angles = self._joint_angles()
        return self._get_observation()

    def _randomize_desired_angles(self):
        self.desired = np.random.rand(1, 7)[0] * 2 - 1

    def _randomize_desired_end_effector_pose(self):
        if self.end_effector_experiment_position:
            self.desired = np.random.rand(1, 3)[0] * 2 - 1
        else:
            self.desired = np.random.rand(1, 7)[0] * 2 - 1

    def get_pose_jacobian(self, poses, jacobians):
        pose_jacobian_dict = {}
        counter = 0
        pose_counter = 0
        jac_counter = 0
        poses = np.array(poses)
        jacobians = np.array(jacobians)
        for i in range(len(joint_names)):
            pose = poses[pose_counter:pose_counter + 3]
            jacobian = np.array([
                jacobians[jac_counter:jac_counter + 7],
                jacobians[jac_counter + 7:jac_counter + 14],
                jacobians[jac_counter + 14:jac_counter + 21],
            ])
            pose_counter += 3
            jac_counter += 21
            pose_jacobian_dict['right' + joint_names[counter]] = [pose, jacobian]
            counter += 1
        return pose_jacobian_dict

    def _get_robot_pose_jacobian_client(self, name):
        rospy.wait_for_service('get_robot_pose_jacobian')
        try:
            get_robot_pose_jacobian = rospy.ServiceProxy('get_robot_pose_jacobian', getRobotPoseAndJacobian,
                                                         persistent=True)
            resp = get_robot_pose_jacobian(name)
            pose_jac_dict = self.get_pose_jacobian(resp.poses, resp.jacobians)
            return pose_jac_dict
        except rospy.ServiceException as e:
            print(e)

    def check_joints_in_box(self, joint_dict):
        keys_to_remove = []
        for joint in joint_dict.keys():
            if self.is_in_box(joint_dict[joint][0]):
               keys_to_remove.append(joint)
        for key in keys_to_remove:
            del joint_dict[key]
        return joint_dict

    def is_in_box(self, endpoint_pose):
        within_box = [curr_pose > lower_pose and curr_pose < higher_pose
            for curr_pose, lower_pose, higher_pose
            in zip(endpoint_pose, box_lows, box_highs)]
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
        if curr_x > box_highs[0]:
            x = -1 * np.exp(np.abs(curr_x - box_highs[0]) * self.temp) * self.safety_force_magnitude
        elif curr_x < box_lows[0]:
            x = np.exp(np.abs(curr_x - box_lows[0]) * self.temp) * self.safety_force_magnitude

        if curr_y > box_highs[1]:
            y = -1 * np.exp(np.abs(curr_y - box_highs[1]) * self.temp) * self.safety_force_magnitude
        elif curr_y < box_lows[1]:
            y = np.exp(np.abs(curr_y - box_lows[1]) * self.temp) * self.safety_force_magnitude

        if curr_z > box_highs[2]:
            z = -1 * np.exp(np.abs(curr_z - box_highs[2]) * self.temp) * self.safety_force_magnitude
        elif curr_z < box_lows[2]:
            z = np.exp(np.abs(curr_z - box_highs[2]) * self.temp) * self.safety_force_magnitude

        return np.array([x, y, z])

    def compute_distances_outside_box(self, pose):
        curr_x = pose[0]
        curr_y = pose[1]
        curr_z = pose[2]
        if(self.is_in_box(pose)):
            x, y, z = 0, 0, 0
        else:
            x, y, z = 0, 0, 0
            if curr_x > box_highs[0]:
                x = np.abs(curr_x - box_highs[0])
            elif curr_x < box_lows[0]:
                x = np.abs(curr_x - box_lows[0])
            if curr_y > box_highs[1]:
                y = np.abs(curr_y - box_highs[1])
            elif curr_y < box_lows[1]:
                y = np.abs(curr_y - box_lows[1])
            if curr_z > box_highs[2]:
                z = np.abs(curr_z - box_highs[2])
            elif curr_z < box_lows[2]:
                z = np.abs(curr_z - box_lows[2])
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
        np.save('training_buffer.npy', self.q)
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
                        desired_orientations.append(observation[28:32])

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
            # obs_torques = []
            # grav_torques = []
            # obsSets = [path["observations"] for path in paths]
            # for obsSet in obsSets:
            #     for observation in obsSet:
            #         obs_torques.append(observation[14:21])
            #         grav_torques.append(observation[24:31])
            #
            # obs_j0 = [torque[0] for torque in obs_torques]
            # obs_j1 = [torque[1] for torque in obs_torques]
            # obs_j2 = [torque[2] for torque in obs_torques]
            # obs_j3 = [torque[3] for torque in obs_torques]
            # obs_j4 = [torque[4] for torque in obs_torques]
            # obs_j5 = [torque[5] for torque in obs_torques]
            # obs_j6 = [torque[6] for torque in obs_torques]
            #
            # grav_j0 = [torque[0] for torque in grav_torques]
            # grav_j1 = [torque[1] for torque in grav_torques]
            # grav_j2 = [torque[2] for torque in grav_torques]
            # grav_j3 = [torque[3] for torque in grav_torques]
            # grav_j4 = [torque[4] for torque in grav_torques]
            # grav_j5 = [torque[5] for torque in grav_torques]
            # grav_j6 = [torque[6] for torque in grav_torques]
            #
            # statistics.update(self._statistics_from_observations(
            #     obs_j0,
            #     stat_prefix,
            #     'Observation Torque: j0'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     obs_j1,
            #     stat_prefix,
            #     'Observation Torque: j1'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     obs_j2,
            #     stat_prefix,
            #     'Observation Torque: j2'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     obs_j3,
            #     stat_prefix,
            #     'Observation Torque: j3'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     obs_j4,
            #     stat_prefix,
            #     'Observation Torque: j4'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     obs_j5,
            #     stat_prefix,
            #     'Observation Torque: j5'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     obs_j6,
            #     stat_prefix,
            #     'Observation Torque: j6'
            # ))
            #
            #
            # statistics.update(self._statistics_from_observations(
            #     grav_j0,
            #     stat_prefix,
            #     'Gravity Torque: j0'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     grav_j1,
            #     stat_prefix,
            #     'Gravity Torque: j1'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     grav_j2,
            #     stat_prefix,
            #     'Gravity Torque: j2'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     grav_j3,
            #     stat_prefix,
            #     'Gravity Torque: j3'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     grav_j4,
            #     stat_prefix,
            #     'Gravity Torque: j4'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     grav_j5,
            #     stat_prefix,
            #     'Gravity Torque: j5'
            # ))
            #
            # statistics.update(self._statistics_from_observations(
            #     grav_j6,
            #     stat_prefix,
            #     'Gravity Torque: j6'
            # ))

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
