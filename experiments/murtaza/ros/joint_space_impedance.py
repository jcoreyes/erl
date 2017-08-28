import rospy
from std_msgs.msg import Empty

import intera_interface
import baxter_interface
import numpy as np

class PDController(object):
    """
    Modified PD Controller for Moving to Neutral

    @param robot: the name of the robot to run the pd controller
    @param limb_name: limb on which to run the pd controller

    """
    def __init__(self, robot="sawyer", limb_name="right"):

        # control parameters
        self._rate = 1000  # Hz
        self._missed_cmds = 20.0  # Missed cycles before triggering timeout

        # create our limb instance
        self.robot = robot
        self._limb_name = limb_name
        if self.robot == "sawyer":
            self._limb = intera_interface.Limb(self._limb_name)
        else:
            self._limb = baxter_interface.Limb(self._limb_name)

        # initialize parameters
        self._springs = dict()
        self._damping = dict()
        self._des_angles = dict()

        # create cuff disable publisher
        cuff_ns = 'robot/limb/' + self._limb_name + '/suppress_cuff_interaction'
        self._pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)

        if self.robot == "sawyer":
            # self._des_angles = {
            #     'right_j1': -1.1778642578125,
            #     'right_j0': 0.0018681640625,
            #     'right_j3': 2.1776962890625,
            #     'right_j2': -0.00246484375,
            #     'right_j6': 3.31884765625,
            #     'right_j4': 0.0015673828125,
            #     'right_j5': 0.5689052734375
            # }
            self._des_angles = {'right_j4': 0.0082041015625, 'right_j5': 0.5668271484375, 'right_j6': 3.318640625, 'right_j1': -1.180951171875, 'right_j2': -0.0023212890625, 'right_j0': -0.0003896484375, 'right_j3': 2.1798525390625}
        else:
            if self._limb_name == "right":
                self._des_angles = {
                    'right_s0': -3.60374974839317e-06,
                    'right_s1': -0.12870475882087096,
                    'right_w0': -0.00016863030052416406,
                    'right_w1': 1.2587168606832257,
                    'right_w2': 3.554180033837895e-06,
                    'right_e0': -7.417427355882467e-06,
                    'right_e1': 0.7519094513387055
                }

            elif self._limb_name == "left":
                self._des_angles = {
                    'left_w0': -0.00016828709045402235,
                    'left_w1': 1.2587240607357284,
                    'left_w2': 3.395214502432964e-06,
                    'left_e0': -7.772197207600584e-06,
                    'left_e1': 0.7519133586749298,
                    'left_s0': -3.857765271675362e-07,
                    'left_s1': -0.13160254316056808
                }


        self.max_stiffness = 20
        self.time_to_maxstiffness = .3  ######### 0.68
        self.t_release = rospy.get_time()

        self._imp_ctrl_is_active = True

        for joint in self._limb.joint_names():
            self._springs[joint] = 30
            self._damping[joint] = 4


    def _set_des_pos(self, des_angles_dict):
        self._des_angles = des_angles_dict

    def adjust_springs(self):
        for joint in list(self._des_angles.keys()):
            t_delta = rospy.get_time() - self.t_release
            if t_delta > 0:
                if t_delta < self.time_to_maxstiffness:
                    self._springs[joint] = t_delta/self.time_to_maxstiffness * self.max_stiffness
                else:
                    self._springs[joint] = self.max_stiffness
            else:
                print("warning t_delta smaller than zero!")

    def _update_forces(self):
        """
        Calculates the current angular difference between the start position
        and the current joint positions applying the joint torque spring forces
        as defined on the dynamic reconfigure server.
        """

        # print self._springs
        self.adjust_springs()

        # disable cuff interaction
        if self._imp_ctrl_is_active:
            self._pub_cuff_disable.publish()

        # create our command dict
        cmd = dict()
        # record current angles/velocities
        cur_pos = self._limb.joint_angles()
        cur_vel = self._limb.joint_velocities()
        # calculate current forces

        for joint in list(self._des_angles.keys()):
            # spring portion
            cmd[joint] = self._springs[joint] * (self._des_angles[joint] -
                                                 cur_pos[joint])
            # damping portion
            cmd[joint] -= self._damping[joint] * cur_vel[joint]

        if self.robot == 'sawyer':
            cmd = np.array(
                [cmd['right_j0'], cmd['right_j1'], cmd['right_j2'], cmd['right_j3'], cmd['right_j4'],
                cmd['right_j5'], cmd['right_j6']])
        else:
            if self._limb_name == "right":
                cmd = np.array(
                    [cmd['right_s0'], cmd['right_s1'], cmd['right_e0'], cmd['right_e1'], cmd['right_w0'],
                     cmd['right_w1'], cmd['right_w2']])
            elif self._limb_name == "left":
                cmd = np.array(
                    [cmd['left_s0'], cmd['left_s1'], cmd['left_e0'], cmd['left_e1'], cmd['left_w0'],
                     cmd['left_w1'], cmd['left_w2']])
        return cmd

