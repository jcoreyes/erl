import rospy
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import baxter_interface as bi
import numpy as np
from cached_property import cached_property
from rllab.envs.base import Env
from rllab.misc import logger
from numpy import linalg
#JACOBIAN STUFF SHOULD GO HERE
from robot_info.srv import *
import random
import ipdb

NUM_JOINTS = 7

"""
These are just ball-parks. For more specific specs, either measure them
and/or see http://sdk.rethinkrobotics.com/wiki/Hardware_Specifications.
"""

###TODO: FIX JOINT_ANGLES

joint_angle_experiment = True
fixed_angle = True
end_effector_experiment_position = False
end_effector_experiment_total = False
fixed_end_effector = False
safety_fixed_angle = False
# safety_limited_end_effector = False

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

END_EFFECTOR_POS_LOW = [0.3404830862298487, -1.2633121086809487, -0.5698485041484043]
END_EFFECTOR_POS_HIGH = [1.1163239572333106, 0.003933425621414761, 0.795699462010194]

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

# safety box:
right_top_left = np.array([0.5593476422885908, -1.2633121086809487, 0.795699462010194])
right_top_right = np.array([1.1163239572333106, 0.003933425621414761, 0.795699462010194])
right_bottom_left = np.array([0.3404830862298487, -0.8305357734786465, -0.569848507615453])
right_bottom_right = np.array([0.6810604337404508, -0.10962952928553238, -0.5698485041484043])

left_top_left = np.array([0.5593476422885908, 1.2633121086809487, 0.795699462010194])
left_top_right = np.array([1.1163239572333106, -0.003933425621414761, 0.795699462010194])
left_bottom_left = np.array([0.3404830862298487, 0.8305357734786465, -0.569848507615453])
left_bottom_right = np.array([0.6810604337404508, 0.10962952928553238, -0.5698485041484043])

#limits for predefined box for safety mode
right_lows = [0.3404830862298487, -1.2633121086809487, -0.5698485041484043]
right_highs = [1.1163239572333106, 0.003933425621414761, 0.795699462010194]

left_lows = [0.3404830862298487, -0.003933425621414761, -0.5698485041484043]
left_highs = [1.1163239572333106, 1.2633121086809487, 0.795699462010194]

#max force that should be applied 
end_effector_force = np.ones(3)

# RIGHT ARM POSE: (AT ZERO JOINT_ANGLES)
# x=0.9048343033476591, y=-1.10782475483212, z=0.3179643218511679

# LEFT ARM POSE: (AT ZERO JOINT_ANGLES)
# position': Point(x=0.9067813662539473, y=1.106112343313852, z=0.31764719868253904)
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
            use_right_arm,
            update_hz=20,
            robot_name='robot',
            action_mode='torque',
            safety_mode='position',
            joint_angle_experiment = True,
            fixed_angle = True,
            end_effector_experiment_position = False,
            end_effector_experiment_total = False,
            fixed_end_effector = False,
            safety_fixed_angle = False,
            safety_limited_end_effector = False,
            delta=10,
            huber=False,
    ):
        Serializable.quick_init(self, locals())
        rospy.init_node('baxter_env', anonymous=True)
        self.rate = rospy.Rate(update_hz)
        self.use_right_arm = use_right_arm
        self.huber = huber
        self.delta = delta
        self.safety_limited_end_effector = safety_limited_end_effector
        #setup the robots arm and gripper
        if(self.use_right_arm):
            self.arm = bi.Limb('right')
            self.arm_joint_names = self.arm.joint_names()
            self.grip = bi.Gripper('right', bi.CHECK_VERSION)
        else:
            self.arm = bi.Limb('left')
            self.arm_joint_names = self.arm.joint_names()
            self.grip = bi.Gripper('left', bi.CHECK_VERSION)

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
        if not self.is_in_box():
            jacobian = self.get_jacobian()
            #implement force adjustment based on which edge was violated!
            torques = jacobian.T @ end_effector_force
            action = action + torques
        
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
        if end_effector_experiment_total:
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

        is_valid = self.is_in_box()

        if joint_angle_experiment:
            #reward is MSE between current joint angles and the desired angles
            reward = -np.mean((self._joint_angles() - self.desired)**2)
            if self.huber:
                a = np.mean(np.abs(self.desired - self._joint_angles()))
                if a <= self.delta:
                    reward = -1/2 * a **2
                else:
                    reward = -1 * self.delta * (a - 1/2 * self.delta)
            
        if end_effector_experiment_position or end_effector_experiment_total:
            #reward is MSE between desired position/orientation and current position/orientation of end_effector
            current_end_effector_pose = self._end_effector_pose()
            reward = -np.mean((current_end_effector_pose - self.desired)**2)
            if self.huber:
                a = np.abs(np.mean(self.desired - current_end_effector_pose))
                if a <= self.delta:
                        reward = -1/2 * a **2
                else:
                    reward = -1 * self.delta * (a- 1/2 * self.delta)


        if not is_valid:
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
        positions = [positions_dict[joint] for joint in self.arm_joint_names]
        velocities = [velocities_dict[joint] for joint in self.arm_joint_names]
        torques = [torques_dict[joint] for joint in self.arm_joint_names]
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
        self.arm.move_to_neutral()

        return self._get_joint_values()

    def _randomize_desired_angles(self):
        self.desired = np.random.rand(1, 7)[0]

    def _randomize_desired_end_effector_pose(self):
        if end_effector_experiment_position:
            self.desired = np.random.rand(1, 3)[0]
        else:
            self.desired = np.random.rand(1, 7)[0]

    def get_jacobian_client(self):
        rospy.wait_for_service('get_jacobian')
        try:
            get_jacobian = rospy.ServiceProxy('get_jacobian', GetJacobian)
            resp = get_jacobian()
            return np.array([resp.jacobianr1, resp.jacobianr2, resp.jacobianr3, resp.jacobianr4, resp.jacobianr5, resp.jacobianr6])
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def get_jacobian(self):
        return self.get_jacobian_client()[:3] #only want rows of jacobian corresponding to the xyz coordinates of end-effector

    def is_in_box(self):
        if safety_fixed_angle or self.safety_limited_end_effector:
            endpoint_pose = self._end_effector_pose()
            if self.use_right_arm:
                within_box = [curr_pose > lower_pose and curr_pose < higher_pose
                    for curr_pose, lower_pose, higher_pose 
                    in zip(endpoint_pose, right_lows, right_highs)]
            else:
                within_box = [curr_pose > lower_pose and curr_pose < higher_pose
                    for curr_pose, lower_pose, higher_pose 
                    in zip(endpoint_pose, left_lows, left_highs)]
            return all(within_box)
        return True

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
                    positions.append(observation[21:24])
                    desired_positions.append(observation[24:27])
                    if end_effector_experiment_total:
                        orientations = np.hstack((orientations, observation[24:28]))
                        desired_orientations = np.hstack((desired_orientations, observation[28:]))

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
                    angles.append(observation[:7])
                    desired_angles.append(observation[21:])

            angles = np.array(angles)
            desired_angles = np.array(desired_angles)
            mean_distance = np.mean(linalg.norm(angles - desired_angles, axis=1))
            logger.record_tabular("Mean Difference from desired angle", mean_distance)

        

    @property
    def horizon(self):
        raise NotImplementedError

    def terminate(self):
        self.reset()

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass
