import math
import time
from collections import OrderedDict
import numpy as np
import rospy
from numpy import linalg
from experiments.murtaza.ros.Sawyer.joint_space_impedance_v2 import PDController
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.core.serializable import Serializable
from railrl.core import logger
from rllab.spaces.box import Box
from sawyer_control.srv import observation
from sawyer_control.msg import actions
from sawyer_control.srv import getRobotPoseAndJacobian
from rllab.envs.base import Env

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

# Testing bounding box for Sawyer on pedestal
box_lows = np.array([-0.4063314307903516, -0.4371988870414967, 0.19114132196594727])
box_highs = np.array([0.5444314339226455, 0.5495988452507109, 0.8264100134638303])

#TODO: figure out where this is being used and why it is _l instead _j
joint_names = [
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
            experiment,
            update_hz=20,
            action_mode='torque',
            safety_box=True,
            reward='huber',
            huber_delta=10,
            safety_force_magnitude=2,
            temperature=1.05,
            safe_reset_length=150,
            reward_magnitude=1,
            use_safety_checks=True,
            wrap_reward_angle_computation=True,
    ):

        Serializable.quick_init(self, locals())
        self.init_rospy(update_hz)

        self.arm_name = 'right'
        self.use_safety_checks = use_safety_checks
        self.wrap_reward_angle_computation = wrap_reward_angle_computation
        self.reward_magnitude = reward_magnitude
        self.safety_box = safety_box
        self.safe_reset_length = safe_reset_length
        self.huber_delta = huber_delta
        self.safety_force_magnitude = safety_force_magnitude
        self.temperature = temperature
        self.PDController = PDController()

        #defaults:
        self.joint_angle_experiment = False
        self.fixed_angle = False
        self.end_effector_experiment_position = False
        self.end_effector_experiment_total = False
        self.fixed_end_effector = False

        self._action_space = Box(
            JOINT_VALUE_LOW[action_mode],
            JOINT_VALUE_HIGH[action_mode]
        )

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

        if reward == 'MSE':
            self.reward_function = self._MSE_reward
        elif reward == 'huber':
            self.reward_function = self._Huber_reward
        else:
            self.reward_function = self._Norm_reward

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
                self.desired = np.array([0.68998028, -0.2285752, 0.3477])

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
                self.desired = np.array(
                    [0.598038329445, -0.110192662364, 0.273337957845, 0.999390065723, 0.0329420607071, 0.00603632837369,
                     -0.00989342758435])
            else:
                self._randomize_desired_end_effector_pose()

        self._observation_space = Box(lows, highs)
        self.get_latest_pose_jacobian_dict()
        self.in_reset = True
        self.amplify=5*np.ones(7)
        self.loss_param = {'delta':0.001, 'c':0.0025}

    @safe
    def _act(self, action):
        if self.safety_box:
            self.get_latest_pose_jacobian_dict()
            truncated_dict = self.check_joints_in_box()
            if len(truncated_dict) > 0:
                forces_dict = self._get_adjustment_forces_per_joint_dict(truncated_dict)
                torques = np.zeros(7)
                for joint in forces_dict:
                    torques = torques + np.dot(truncated_dict[joint][1].T, forces_dict[joint]).T
                action = action + torques
        if self.in_reset:
            np.clip(action, -4, 4, out=action)
        if not self.in_reset:
            action = self.amplify * action
            action = np.clip(np.asarray(action),-MAX_TORQUES, MAX_TORQUES)

        self.send_action(action)
        self.rate.sleep()
        return action

    def _reset_within_threshold(self):
        desired_neutral = np.array([
            6.28115601e+00,
            5.10141089e+00,
            6.28014234e+00,
            2.17755176e+00,
            9.48242187e-04,
            5.73669922e-01,
            3.31514160e+00
        ])
        desired_neutral = (desired_neutral)
        actual_neutral = (self._joint_angles())
        errors = self.compute_angle_difference(desired_neutral, actual_neutral)
        ERROR_THRESHOLD = .1*np.ones(7)
        is_within_threshold = (errors < ERROR_THRESHOLD).all()
        return is_within_threshold

    def _wrap_angles(self, angles):
        return angles % (2*np.pi)

    def _joint_angles(self):
        angles, _, _, _ = self.request_observation()
        angles = np.array(angles)
        return angles

    def _end_effector_pose(self):
        _, _, _, endpoint_pose = self.request_observation()
        if self.end_effector_experiment_total:
            return np.array(endpoint_pose)
        else:
            x, y, z, _, _, _, _ = endpoint_pose
            return np.array([x, y, z])

    def _MSE_reward(self, differences):
        reward = -np.mean(differences**2)
        return reward

    def _Huber_reward(self, differences):
        a = np.abs(np.mean(differences))
        if a <= self.huber_delta:
            reward = -1 / 2 * a ** 2 * self.reward_magnitude
        else:
            reward = -1 * self.huber_delta * (a - 1 / 2 * self.huber_delta) * self.reward_magnitude
        return reward

    def _Norm_reward(self, differences):
        return np.linalg.norm(differences)

    def compute_angle_difference(self, angles1, angles2):
        self._wrap_angles(angles1)
        self._wrap_angles(angles2)
        deltas = np.abs(angles1 - angles2)
        differences = np.minimum(2 * np.pi - deltas, deltas)
        return differences

    def step(self, action, task='reaching'):
        self.nan_check(action)
        actual_commanded_action = self._act(action)
        observation = self._get_observation()
        reward = self.rewards(action, task)

        if self.use_safety_checks:
            out_of_box = self.safety_box_check()
            high_torque = self.high_torque_check(actual_commanded_action)
            unexpected_velocity = self.unexpected_velocity_check()
            unexpected_torque = self.unexpected_torque_check()
            done = out_of_box or high_torque or unexpected_velocity or unexpected_torque
        else:
            done = False
        info = {}
        return observation, reward, done, info

    def rewards(self, action, task='reaching'):
        if task == 'lego':
            current = self._end_effector_pose()
            pos_diff = self.desired[:3] - current[:3]
            angle_diff = self.compute_angle_difference(self.desired[3:7], current[3:7])
            reward = self._Lorentz_reward(pos_diff, angle_diff, action)
        else:
            if self.joint_angle_experiment:
                current = self._joint_angles()
                differences = self.compute_angle_difference(current, self.desired)
                reward = self.reward_function(differences)

            elif self.end_effector_experiment_position or self.end_effector_experiment_total:
                current = self._end_effector_pose()
                # reward = -1*np.linalg.norm(self.desired-current) * self.reward_magnitude
                differences = self.desired-current
                reward = self.reward_function(differences)
        return reward

    def safety_box_check(self):
        # TODO: tune this check
        self.get_latest_pose_jacobian_dict()
        truncated_dict = self.check_joints_in_box()
        terminate_episode = False
        if len(truncated_dict) > 0:
            for joint in truncated_dict.keys():
                dist = self._compute_joint_distance_outside_box(truncated_dict[joint][0])
                if dist > .19:
                    if not self.in_reset:
                        print('safety box failure during train/eval: ', joint, dist)
                        terminate_episode = True
                    else:
                        raise EnvironmentError('safety box failure during reset: ', joint, dist)
        return terminate_episode

    def jacobian_check(self):
        ee_jac = self.pose_jacobian_dict['right_hand'](1)
        if np.linalg.det(ee_jac) == 0:
            self._act(self._randomize_desired_angles())

    def unexpected_torque_check(self):
        #TODO: redesign this check
        #we care about the torque that was observed to make sure it hasn't gone too high
        new_torques = self.get_observed_torques_minus_gravity()
        if not self.in_reset:
            ERROR_THRESHOLD = np.array([25, 25, 25, 25, 666, 666, 10])
            is_peaks = (np.abs(new_torques) > ERROR_THRESHOLD).any()
            if is_peaks:
                print('unexpected_torque during train/eval: ', new_torques)
                return True
        else:
            ERROR_THRESHOLD = np.array([25, 25, 25, 30, 666, 666, 10])
            is_peaks = (np.abs(new_torques) > ERROR_THRESHOLD).any()
            if is_peaks:
                raise EnvironmentError('unexpected torques during reset: ', new_torques)
        return False

    def unexpected_velocity_check(self):
        #TODO: tune this check
        _, velocities, _, _ = self.request_observation()
        velocities = np.array(velocities)
        ERROR_THRESHOLD = 5 * np.ones(7)
        is_peaks = (np.abs(velocities) > ERROR_THRESHOLD).any()
        if is_peaks:
            print('unexpected_velocities during train/eval: ', velocities)
            if not self.in_reset:
                return True
            else:
                raise EnvironmentError('unexpected velocities during reset: ', velocities)
        return False

    def high_torque_check(self, commanded_torques):
        # TODO: tune this check
        new_torques = np.abs(commanded_torques)
        current_angles = self._joint_angles()
        position_deltas = np.abs(current_angles - self.previous_angles)
        DELTA_THRESHOLD = .05 * np.ones(7)
        ERROR_THRESHOLD = [11, 15, 15, 15, 666, 666, 10]
        violation = False
        for i in range(len(new_torques)):
            if new_torques[i] > ERROR_THRESHOLD[i] and position_deltas[i] < DELTA_THRESHOLD[i]:
                violation=True
                print("violating joint:", i)
        if violation:
            print('high_torque:', new_torques)
            print('positions', position_deltas)
            if not self.in_reset:
                return True
            else:
                raise EnvironmentError('ERROR: Applying large torques and not moving')
        self.previous_angles = current_angles
        return False

    def nan_check(self, action):
        for val in action:
            if math.isnan(val):
                raise EnvironmentError('ERROR: NaN action attempted')

    def _get_observation(self):
        angles = self._joint_angles()
        _, velocities, torques, _ = self.request_observation()
        velocities = np.array(velocities)
        torques = np.array(torques)
        endpoint_pose = self._end_effector_pose()

        temp = np.hstack((
            angles,
            velocities,
            torques,
            endpoint_pose,
            self.desired
        ))
        return temp

    def _safe_move_to_neutral(self):
        for i in range(self.safe_reset_length):
            cur_pos, cur_vel, _, _ = self.request_observation()
            torques = self.PDController._compute_pd_forces(cur_pos, cur_vel)
            actual_commanded_actions = self._act(torques)
            curr_time = time.time()
            self.init_delay = curr_time
            if self.previous_angles_reset_check():
                break
            if self.use_safety_checks:
                self.safety_box_check()
                self.unexpected_torque_check()
                self.high_torque_check(actual_commanded_actions)
                self.unexpected_velocity_check()

    def previous_angles_reset_check(self):
        # TODO: tune this check so that reset cuts off early but at the right time
        close_to_desired_reset_pos = self._reset_within_threshold()
        _, velocities, _, _ = self.request_observation()
        velocities = np.abs(np.array(velocities))
        VELOCITY_THRESHOLD = .002 * np.ones(7)
        no_velocity = (velocities < VELOCITY_THRESHOLD).all()
        return close_to_desired_reset_pos and no_velocity

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.in_reset = True
        self.previous_angles = self._joint_angles()

        if not self.fixed_angle and self.joint_angle_experiment:
            self._randomize_desired_angles()
        elif not self.fixed_end_effector and not self.end_effector_experiment_position and not self.end_effector_experiment_total:
            self._randomize_desired_end_effector_pose()

        self._safe_move_to_neutral()
        self.previous_angles = self._joint_angles()
        self.in_reset = False
        return self._get_observation()

    def _randomize_desired_angles(self):
        self.desired = np.random.rand(1, 7)[0] * 2 - 1

    def _randomize_desired_end_effector_pose(self):
        if self.end_effector_experiment_position:
            self.desired = np.random.uniform(box_lows, box_highs, size=(1, 3))[0]
        else:
            self.desired = np.random.uniform(box_lows, box_highs, size=(1, 7))[0]

    def get_latest_pose_jacobian_dict(self):
        self.pose_jacobian_dict = self._get_robot_pose_jacobian_client('right')

    def _get_robot_pose_jacobian_client(self, name):
        rospy.wait_for_service('get_robot_pose_jacobian')
        try:
            get_robot_pose_jacobian = rospy.ServiceProxy('get_robot_pose_jacobian', getRobotPoseAndJacobian,
                                                         persistent=True)
            resp = get_robot_pose_jacobian(name)
            pose_jac_dict = self.get_pose_jacobian_dict(resp.poses, resp.jacobians)
            return pose_jac_dict
        except rospy.ServiceException as e:
            print(e)

    def get_pose_jacobian_dict(self, poses, jacobians):
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

    def _get_positions_from_pose_jacobian_dict(self):
        poses = []
        for joint in self.pose_jacobian_dict.keys():
            poses.append(self.pose_jacobian_dict[joint][0])
        return np.array(poses)

    def check_joints_in_box(self):
        joint_dict = self.pose_jacobian_dict.copy()
        keys_to_remove = []
        for joint in joint_dict.keys():
            if self._pose_in_box(joint_dict[joint][0]):
                keys_to_remove.append(joint)
        for key in keys_to_remove:
            del joint_dict[key]
        return joint_dict

    def _pose_in_box(self, pose):
        within_box = [curr_pose > lower_pose and curr_pose < higher_pose
                      for curr_pose, lower_pose, higher_pose
                      in zip(pose, box_lows, box_highs)]
        return all(within_box)

    def _get_adjustment_forces_per_joint_dict(self, joint_dict):
        forces_dict = {}
        for joint in joint_dict:
            force = self._get_adjustment_force_from_pose(joint_dict[joint][0])
            forces_dict[joint] = force
        return forces_dict

    def _get_adjustment_force_from_pose(self, pose):
        x, y, z = 0, 0, 0

        curr_x = pose[0]
        curr_y = pose[1]
        curr_z = pose[2]
        if curr_x > box_highs[0]:
            x = -1 * np.exp(np.abs(curr_x - box_highs[0]) * self.temperature) * self.safety_force_magnitude
        elif curr_x < box_lows[0]:
            x = np.exp(np.abs(curr_x - box_lows[0]) * self.temperature) * self.safety_force_magnitude

        if curr_y > box_highs[1]:
            y = -1 * np.exp(np.abs(curr_y - box_highs[1]) * self.temperature) * self.safety_force_magnitude
        elif curr_y < box_lows[1]:
            y = np.exp(np.abs(curr_y - box_lows[1]) * self.temperature) * self.safety_force_magnitude

        if curr_z > box_highs[2]:
            z = -1 * np.exp(np.abs(curr_z - box_highs[2]) * self.temperature) * self.safety_force_magnitude
        elif curr_z < box_lows[2]:
            z = np.exp(np.abs(curr_z - box_highs[2]) * self.temperature) * self.safety_force_magnitude
        return np.array([x, y, z])

    def _compute_joint_distance_outside_box(self, pose):
        curr_x = pose[0]
        curr_y = pose[1]
        curr_z = pose[2]
        if(self._pose_in_box(pose)):
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

    def log_diagnostics(self, paths):
        statistics = OrderedDict()
        stat_prefix = 'Test'
        if self.end_effector_experiment_total or self.end_effector_experiment_position:
            obsSets = [path["observations"] for path in paths]
            positions = []
            desired_positions = []
            distances = []
            if self.end_effector_experiment_total:
                orientations = []
                desired_orientations = []
            last_n_distances = []
            final_counter = 0
            final_positions = []
            final_desired_positions = []
            for obsSet in obsSets:
                for observation in obsSet:
                    pos = np.array(observation[21:24])
                    des = np.array(observation[24:27])
                    distances.append(np.linalg.norm(pos - des))
                    positions.append(pos)
                    desired_positions.append(des)
                    for observation in obsSet[len(obsSet)-10:len(obsSet)]:
                            pos = np.array(observation[21:24])
                            des = np.array(observation[24:27])
                            last_n_distances.append(np.linalg.norm(pos - des))

                    if self.end_effector_experiment_total:
                        orientations.append(observation[24:28])
                        desired_orientations.append(observation[28:32])

                if self.end_effector_experiment_position:
                    final_counter += len(obsSet)
                    final_positions.append(obsSet[-1][21:24])
                    final_desired_positions.append(obsSet[-1][24:27])

            statistics.update(self._statistics_from_observations(
                distances,
                stat_prefix,
                'Distance from Desired End Effector Position'
            ))

            statistics.update(self._statistics_from_observations(
                last_n_distances,
                stat_prefix,
                'Last N Step Distance from Desired End Effector Position'
            ))

            if self.end_effector_experiment_total:
                orientations_distance = linalg.norm(orientations-desired_orientations, axis=1)
                statistics.update(self._statistics_from_observations(
                    orientations_distance,
                    stat_prefix,
                    'Distance from Desired End Effector Orientation'
                ))

            final_positions = np.array(final_positions)
            final_desired_positions = np.array(final_desired_positions)
            final_position_distances = linalg.norm(final_positions - final_desired_positions, axis=1)
            statistics.update(self._statistics_from_observations(
                final_position_distances,
                stat_prefix,
                'Final Distance from Desired End Effector Position'
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

            differences = np.array([self.compute_angle_difference(angle_obs, desired_angle_obs)
                                    for angle_obs, desired_angle_obs in zip(angles, desired_angles)])
            angle_differences = np.mean(differences, axis=1)
            distances_outside_box = np.array([self._compute_joint_distance_outside_box(pose) for pose in positions])
            return [angle_differences, distances_outside_box]


    def _statistics_from_observations(self, observation, stat_prefix, log_title):
        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            '{} {}'.format(stat_prefix, log_title),
            observation,
        ))

        return statistics


    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def init_rospy(self, update_hz):
        rospy.init_node('sawyer_env', anonymous=True)
        self.action_publisher = rospy.Publisher('actions_publisher', actions, queue_size=10)
        self.rate = rospy.Rate(update_hz)

    def send_action(self, action):
        self.action_publisher.publish(action)

    def request_observation(self):
        rospy.wait_for_service('observations')
        try:
            request = rospy.ServiceProxy('observations', observation, persistent=True)
            obs = request()
            return (
                    obs.angles,
                    obs.velocities,
                    obs.torques,
                    obs.endpoint_pose
            )
        except rospy.ServiceException as e:
            print(e)

    @property
    def horizon(self):
        raise NotImplementedError

    def terminate(self):
        self.reset()
