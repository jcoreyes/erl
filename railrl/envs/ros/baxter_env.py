import rospy
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import baxter_interface as bi
import numpy as np
from cached_property import cached_property
from rllab.envs.base import Env
from rllab.misc import logger
from numpy import linalg
import ipdb

NUM_JOINTS = 7

"""
These are just ball-parks. For more specific specs, either measure them
and/or see http://sdk.rethinkrobotics.com/wiki/Hardware_Specifications.
"""
joint_angle_experiment = False
fixed_angle = False
end_effector_experiment_position = True
end_effector_experiment_total = False
fixed_end_effector = False
safety_fixed_angle = False
position_resetting = False
number_fixed_angles = 2

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
END_EFFECTOR_POS_LOW = -1*np.ones(3)
END_EFFECTOR_POS_HIGH = np.ones(3)

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
            robot_name='robot',
            action_mode='torque',
            safety_mode='position'
    ):
        Serializable.quick_init(self, locals())
        rospy.init_node('baxter_env', anonymous=True)
        self.rate = rospy.Rate(update_hz)

        #setup the robots arm and gripper
        self.right_arm = bi.Limb('right', robot_name=robot_name)
        self.right_joint_names = self.right_arm.joint_names()
        self.right_grip = bi.Gripper('right', robot_name=robot_name, bi.CHECK_VERSION)

        #create a dictionary whose values are functions that set the appropriate values
        action_mode_dict = {
            'position': self.right_arm.set_joint_positions, 
            'velocity': self.right_arm.set_joint_velocities,
            'torque': self.right_arm.set_joint_torques,
        }

        #create a dictionary whose values are functions that return the appropriate values
        observation_mode_dict = {
            'position': self.right_arm.joint_angles,
            'velocity': self.right_arm.joint_velocities,
            'torque': self.right_arm.joint_efforts,
        }

        self._set_joint_values = action_mode_dict[action_mode]
        self._get_joint_to_value_func_list = list(observation_mode_dict.values())

        #set up safety mode components (fixing a certain number of joint angles)
        if safety_fixed_angle:

            self._set_joint_values = action_mode_dict
            self.action_mode = action_mode
            self.safety_mode = safety_mode
            self.initial_positions = self._joint_angles()[:number_fixed_angles]
            self._action_space = Box(
                JOINT_VALUE_LOW[action_mode][number_fixed_angles:], 
                JOINT_VALUE_HIGH[action_mode][number_fixed_angles:]
                )

        else:
            self._action_space = Box(
                JOINT_VALUE_LOW[action_mode],
                JOINT_VALUE_HIGH[action_mode]
                )

        #set up lows and highs for observation space based on which experiment we are running
        #additionally set up the desired angle as well
        if joint_angle_experiment:
            lows = np.hstack((
                JOINT_VALUE_LOW['position'], 
                JOINT_VALUE_LOW['velocity'], 
                JOINT_VALUE_LOW['torque'], 
                JOINT_VALUE_LOW['position'],
            ))

            highs = np.hstack((
                JOINT_VALUE_HIGH['position'], 
                JOINT_VALUE_HIGH['velocity'],
                JOINT_VALUE_HIGH['torque'], 
                JOINT_VALUE_HIGH['position'],
            ))

            if fixed_angle:
                self.desired = np.zeros(NUM_JOINTS) 
            else:
                self._randomize_desired_angles() 

        elif end_effector_experiment_position:
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

            if fixed_end_effector:
                self.desired = np.array([
                    0.1485434521312332, 
                    -0.43227588084273644, 
                    -0.7116727296474704
                ])

            else:
                self._randomize_desired_end_effector_pose()

        elif end_effector_experiment_total:
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
            
            if fixed_end_effector:
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
        if safety_fixed_angle:
            if(position_resetting):
                fixed_names = self.right_joint_names[:number_fixed_angles]
                fixed_positions = dict(zip(fixed_names, self.initial_positions))
                self._set_joint_values[self.safety_mode](fixed_positions)
            action_names = self.right_joint_names[number_fixed_angles:]
            actions = dict(zip(action_names, action))
            self._set_joint_values[self.action_mode](actions)

        else:
            joint_to_values = dict(zip(self.right_joint_names, action))
            self._set_joint_values(joint_to_values)
    	
        self.rate.sleep()

    def _joint_angles(self):
        joint_to_angles = self.right_arm.joint_angles()
        return np.array([
            joint_to_angles[joint] for joint in self.right_joint_names
        ])

    def _end_effector_pose(self):
        state_dict = self.right_arm.endpoint_pose()
        pos = state_dict['position']
        if end_effector_experiment_total or safety_fixed_angle:
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
        """
        :param deltas: a change joint angles
        """
        self._act(action)
        observation = self._get_joint_values()

        is_valid = True
        if safety_fixed_angle:
            endpoint_pose = self._end_effector_pose()
            lower_endpoint_pose = np.hstack((
                END_EFFECTOR_VALUE_LOW['position'], 
                END_EFFECTOR_VALUE_LOW['angle'],
            ))
            higher_endpoint_pose = np.hstack((
                END_EFFECTOR_VALUE_HIGH['position'], 
                END_EFFECTOR_VALUE_HIGH['angle']
            ))
            within_box = [curr_pose > lower_pose or curr_pose < higher_pose
                for curr_pose, lower_pose, higher_pose 
                in zip(endpoint_pose, lower_endpoint_pose, higher_endpoint_pose)]
            is_valid = all(within_box)

        if joint_angle_experiment:
            #reward is MSE between current joint angles and the desired angles
            clipped_joint_angles = self._joint_angles()[number_fixed_angles]
            reward = -np.mean((clipped_joint_angles - self.desired[number_fixed_angles:])**2)
            
        if end_effector_experiment_position or end_effector_experiment_total:
            #reward is MSE between desired position/orientation and current position/orientation of end_effector
            current_end_effector_pose = self._end_effector_pose()
            reward = -np.mean((current_end_effector_pose - self.desired)**2)

        if not is_valid:
            reward = -10000
            done = True
        else:
            done = False

        info = {}
        return observation, reward, done, info

    def _get_joint_values(self):
        # joint_values_dict = self._get_joint_to_value_dict()
        positions_dict = self._get_joint_to_value_func_list[0]()
        velocities_dict = self._get_joint_to_value_func_list[1]()
        torques_dict = self._get_joint_to_value_func_list[2]()
        positions = [positions_dict[joint] for joint in self.right_joint_names]
        velocities = [velocities_dict[joint] for joint in self.right_joint_names]
        torques = [torques_dict[joint] for joint in self.right_joint_names]
        temp = velocities + torques

        if end_effector_experiment_position or end_effector_experiment_total:
            temp = np.hstack((temp, self._end_effector_pose()))

        temp = np.hstack((positions, temp, self.desired))

        return temp

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        if joint_angle_experiment and not fixed_angle:
            self._randomize_desired_angles()
        elif end_effector_experiment_position or end_effector_experiment_total and not fixed_end_effector:
            self._randomize_desired_end_effector_pose()

        self.right_arm.move_to_neutral()

        if safety_fixed_angle:
            self.initial_positions = self._joint_angles()[:number_fixed_angles]

        return self._get_joint_values()

    def _randomize_desired_angles(self):
        self.desired = np.random.rand(1, 7)

    def _randomize_desired_end_effector_pose(self):
        if end_effector_experiment_position:
            self.desired = np.random.rand(1, 3)[0]
        else:
            self.desired = np.random.rand(1, 7)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def render(self):
        pass

    def log_diagnostics(self, paths):
        if end_effector_experiment_total or end_effector_experiment_position:
            obsSets = [path["observations"] for path in paths]
            positions = []
            desired_positions = []
            if end_effector_experiment_total:
                orientations = []
                desired_orientations = []
            for obsSet in obsSets:
                for observation in obsSet:
                    # positions = np.hstack((positions, observation[21:24]))
                    # desired_positions = np.hstack((desired_positions, observation[24:27]))
                    positions.append(observation[21:24])
                    desired_positions.append(observation[24:27])
                    # ipdb.set_trace()
                    
                    if end_effector_experiment_total:
                        orientations = np.hstack((orientations, observation[24:28]))
                        desired_orientations = np.hstack((desired_orientations, observation[28:]))

            # mean_distance = np.sum(positions - desired_positions)
            positions = np.array(positions)
            desired_positions = np.array(desired_positions)
            mean_distance = np.mean(linalg.norm(positions - desired_positions, axis=1))
            logger.record_tabular("Mean Distance from desired end-effector position", mean_distance)


            if end_effector_experiment_total:
                mean_orientation_difference = np.mean(orientations - desired_orientations)
                logger.record_tabular("Mean Orientation difference from desired end-effector position", mean_orientation_difference)

        if joint_angle_experiment:
            obsSets = [path["observations"] for path in paths]
            angles = []
            desired_angles = []
            for obsSet in obsSets:
                for observation in obsSet:
                    angles = np.hstack((angles, observation[:7]))
                    desired_angles = np.hstack((desired_angles, observation[21:]))

            mean_difference = np.mean(angles - desired_angles)
            logger.record_tabular("Mean Difference from desired angle", mean_difference)

        

    @property
    def horizon(self):
        raise NotImplementedError

    def terminate(self):
        self.reset()

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass
