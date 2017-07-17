import rospy
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import intera_interface as ii
import numpy as np
from rllab.envs.base import Env
from rllab.misc import logger
from numpy import linalg
from robot_info.srv import *
import ipdb

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

JOINT_TORQUE_HIGH = 0.5*np.ones(7)
JOINT_TORQUE_LOW = -0.5*np.ones(7)

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

ee_box_highs = [
    0.7175958839273338,
    0.3464466563902636,
    0.7659791453416877,
]

ee_box_lows = [
    0.1628008448954529,
    -0.33786487626917794,
    0.20084391863426093,
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
            use_gripper=False,
            action_mode='torque',
            remove_action=False,
            safety_end_effector_box=False,
            reward_function='huber',
            huber_delta=10,
            safety_box_magnitude=2,
            safety_box_temp=1.05,
    ):

        Serializable.quick_init(self, locals())
        rospy.init_node('sawyer_env', anonymous=True)
        self.rate = rospy.Rate(update_hz)

        #defaults:
        self.use_gripper = use_gripper
        self.joint_angle_experiment = False
        self.fixed_angle = False
        self.end_effector_experiment_position = False
        self.end_effector_experiment_total = False
        self.fixed_end_effector = False
        self.safety_end_effector_box = False

        if experiment == experiments[0]:
            self.joint_angle_experiment=True
        elif experiment == experiments[1]:
            self.joint_angle_experiment=True
            self.fixed_angle=False
        elif experiment == experiments[2]:
            self.end_effector_experiment_position=True
        elif experiment == experiments[3]:
            self.end_effector_experiment_position=False
            self.fixed_end_effector = False
        elif experiment == experiments[4]:
            self.end_effector_experiment_total=True
        elif experiment == experiments[5]:
            self.end_effector_experiment_total = True
            self.fixed_end_effector=False
        
        self.safety_end_effector_box = safety_end_effector_box
        self.remove_action = remove_action

        if reward_function == 'MSE':
            self.MSE = True
            self.huber=False
        elif reward_function == 'huber':
            self.huber = True
            self.MSE = False

        self.huber_delta = huber_delta
        self.safety_box_magnitude = safety_box_magnitude
        self.safety_box_temp = safety_box_temp

        self.arm = ii.Limb('right')
        self.arm_joint_names = self.arm.joint_names()

        if self.use_gripper:
            self.grip = bi.Gripper('right', bi.CHECK_VERSION)

        #create a dictionary whose values are functions that set the appropriate values
        action_mode_dict = {
            'position': self.arm.set_joint_positions, 
            'velocity': self.arm.set_joint_velocities,
            'torque': self.arm.set_joint_torques,
        }

        #create a dictionary whose values are functions that return the appropriate values
        observation_mode_dict = {
            'position': self.arm.joint_angles,
            'velocity': self.arm.joint_velocities,
            'torque': self.arm.joint_efforts,
        }

        self._set_joint_values = action_mode_dict[action_mode]
        self._get_joint_to_value_func_list = list(observation_mode_dict.values())

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
                #self.desired = np.zeros(NUM_JOINTS)
                angle_dict = {
                    'right_j6': 1.6192763671875,
                    'right_j5': 1.8405947265625,
                    'right_j4': 1.7157666015625,
                    'right_j3': 1.95903125,
                    'right_j2': -1.7705771484375,
                    'right_j1': 0.13664453125,
                    'right_j0': 0.143740234375
                }
                angles = list(angle_dict.values())
                angles.reverse()
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
                self.desired = np.array([
                    0.5217189571796944,
                    -0.10860045563531961,
                    0.3957934159711215,
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
        if self.safety_end_effector_box and not self.is_in_box(self._end_effector_pose()):
            jacobian = self.get_jacobian()
            #ipdb.set_trace()
            end_effector_force = self.get_adjustment_force()
            torques = np.dot(jacobian.T, end_effector_force).T
            if self.remove_action:
                action = torques
            else:
                action = action + torques
            # ipdb.set_trace()

        # np.clip(action, -.5, .5, out=action)
        joint_to_values = dict(zip(self.arm_joint_names, action))
        self._set_joint_values(joint_to_values)
        self.rate.sleep()

    def _joint_angles(self):
        joint_to_angles = self.arm.joint_angles()
        return np.array([
            joint_to_angles[joint] for joint in self.arm_joint_names
        ])

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

    def step(self, action):
        self.terminate = False
        self._act(action)
        observation = self._get_joint_values()

        if self.joint_angle_experiment:
            if self.MSE:
                reward = -np.mean((self._joint_angles() - self.desired)**2)
            elif self.huber:
                a = np.mean(np.abs(self.desired - self._joint_angles()))
                if a <= self.huber_delta:
                    reward = -1/2 * a **2
                else:
                    reward = -1 * self.huber_delta * (a - 1/2 * self.huber_delta)
            
        if self.end_effector_experiment_position or self.end_effector_experiment_total:
            current_end_effector_pose = self._end_effector_pose()
            if self.MSE:
                reward = -np.mean((current_end_effector_pose - self.desired)**2)
            elif self.huber:
                a = np.abs(np.mean(self.desired - current_end_effector_pose))
                if a <= self.huber_delta:
                        reward = -1/2 * a **2
                else:
                    reward = -1 * self.huber_delta * (a- 1/2 * self.huber_delta)

        done = self.terminate
        info = {}
        return observation, reward, done, info

    def _get_joint_values(self):
        # joint_values_dict = self._get_joint_to_value_dict()
        positions_dict = self._get_joint_to_value_func_list[0]()
        velocities_dict = self._get_joint_to_value_func_list[1]()
        torques_dict = self._get_joint_to_value_func_list[2]()
        positions = [positions_dict[joint] for joint in self.arm_joint_names]
        velocities = [velocities_dict[joint] for joint in self.arm_joint_names]
        torques = [torques_dict[joint] for joint in self.arm_joint_names]
        safety_box_temp = positions + velocities + torques
        safety_box_temp = np.hstack((safety_box_temp, self._end_effector_pose()))
        safety_box_temp = np.hstack((safety_box_temp, self.desired))
        return safety_box_temp

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
        # #ipdb.set_trace()
        return self._get_joint_values()

    def _randomize_desired_angles(self):
        self.desired = np.random.rand(1, 7)[0]

    def _randomize_desired_end_effector_pose(self):
        if self.end_effector_experiment_position:
            self.desired = np.random.rand(1, 3)[0]
        else:
            self.desired = np.random.rand(1, 7)[0]

    def get_jacobian_client(self):
        rospy.wait_for_service('get_jacobian')
        try:
            get_jacobian = rospy.ServiceProxy('get_jacobian', GetJacobian)
            resp = get_jacobian('right')
            #ipdb.set_trace()
            return np.array([resp.jacobianr1,
                             resp.jacobianr2,
                             resp.jacobianr3,
                             resp.jacobianr4,
                             resp.jacobianr5,
                             resp.jacobianr6])
        except Exception as e:
            # self.terminate = True
            #ipdb.set_trace()
            return np.zeros((6, 7))

    def get_jacobian(self):
        return self.get_jacobian_client()[:3]
        
    def is_in_box(self, endpoint_pose):

        is_in_box = True
        if self.safety_end_effector_box:
            within_box = [curr_pose > lower_pose and curr_pose < higher_pose
                for curr_pose, lower_pose, higher_pose
                in zip(endpoint_pose, ee_box_lows, ee_box_highs)]
            is_in_box = all(within_box)
        return is_in_box

    def get_adjustment_force(self):
        x, y, z = 0, 0, 0
        endpoint_pose = self._end_effector_pose()
        curr_x = endpoint_pose[0]
        curr_y = endpoint_pose[1]
        curr_z = endpoint_pose[2]

        if curr_x > ee_box_highs[0]:
            x = -1 * np.exp(np.abs(curr_x - ee_box_highs[0]) * self.safety_box_temp) * self.safety_box_magnitude
        elif curr_x < ee_box_lows[0]:
            x = np.exp(np.abs(curr_x - ee_box_lows[0]) * self.safety_box_temp) * self.safety_box_magnitude

        if curr_y > ee_box_highs[1]:
             y = -1 * np.exp(np.abs(curr_y - ee_box_highs[1]) * self.safety_box_temp) * self.safety_box_magnitude
        elif curr_y < ee_box_lows[1]:
            y = np.exp(np.abs(curr_y - ee_box_lows[1]) * self.safety_box_temp) * self.safety_box_magnitude

        if curr_z > ee_box_highs[2]:
            z = -1 * np.exp(np.abs(curr_z - ee_box_highs[2]) * self.safety_box_temp) * self.safety_box_magnitude
        elif curr_z < ee_box_lows[2]:
            z = np.exp(np.abs(curr_z - ee_box_highs[2]) * self.safety_box_temp) * self.safety_box_magnitude

        return np.array([x, y, z])

    def compute_mean_distance_outside_box(self, pose):
        curr_x = pose[0]
        curr_y = pose[1]
        curr_z = pose[2]
        x, y, z = 0, 0, 0

        if curr_x > ee_box_highs[0]:
            x = np.abs(curr_x - ee_box_highs[0])
        elif curr_x < ee_box_lows[0]:
            x = np.abs(curr_x - ee_box_lows[0])

        if curr_y > ee_box_highs[1]:
            y = np.abs(curr_y - ee_box_highs[1])
        elif curr_y < ee_box_lows[1]:
            y = np.abs(curr_y - ee_box_lows[1])

        if curr_z > ee_box_highs[2]:
            z = np.abs(curr_z - ee_box_highs[2])
        elif curr_z < ee_box_lows[2]:
            z = np.abs(curr_z - ee_box_lows[2])

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
            mean_distance_from_desired_ee_pose = np.mean(linalg.norm(positions - desired_positions, axis=1))
            logger.record_tabular("Mean Distance from desired end-effector position",
                                  mean_distance_from_desired_ee_pose)

            if self.safety_end_effector_box:
                mean_distance_outside_box = np.mean([self.compute_mean_distance_outside_box(pose) for pose in positions])
                logger.record_tabular("Mean Distance Outside Box", mean_distance_outside_box)

            if self.end_effector_experiment_total:
                mean_orientation_difference = np.mean(linalg.norm(orientations-desired_orientations), axis=1)
                logger.record_tabular("Mean Orientation difference from desired end-effector position",
                                      mean_orientation_difference)

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

            if self.safety_end_effector_box:
                mean_distance_outside_box = np.mean(
                    [self.compute_mean_distance_outside_box(pose) for pose in positions if not self.is_in_box(pose)])
                logger.record_tabular("Mean Distance Outside Box", mean_distance_outside_box)
    @property
    def horizon(self):
        raise NotImplementedError

    def terminate(self):
        self.reset()

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass
