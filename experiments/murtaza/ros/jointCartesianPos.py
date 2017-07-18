from urdf_parser_py.urdf import URDF 
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
import baxter_interface as ii
import rospy
import numpy as np 
robot = URDF.from_parameter_server(key='robot_description')
from pykdl_utils.kdl_kinematics import KDLKinematics
base_link = 'base'
end_link = 'right_gripper'
kdl_kin = KDLKinematics(robot, base_link, end_link)
rospy.init_node('bax')
arm = ii.Limb('right')
q = arm.joint_angles()
q = [q['right_s0'], q['right_s1'], q['right_e0'], q['right_e1'], q['right_w0'], q['right_w1'], q['right_w2']]
pose = kdl_kin.forward(q, 'base', 'right_gripper')
pose = kdl_kin.forward(q, end_link='right_upper_forearm')
pose = np.squeeze(np.asarray(pose))
pose = [pose[0][3], pose[1][3], pose[2][3]]
print pose
