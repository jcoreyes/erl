import intera_interface as ii
import rospy
import numpy as np
rospy.init_node('right')
rate = rospy.Rate(20)
arm = ii.Limb('right')
arm.move_to_neutral()
# pos = arm.endpoint_pose()['position']
# print([pos.x, pos.y, pos.z])
# for _ in range(100):
#     action = np.ones(7)
#     joint_to_values = dict(zip(arm.joint_names(), action))
#     arm.set_joint_torques(joint_to_values)
#     rate.sleep()
# ja = arm.joint_angles()
# ja = np.array([ja['right_j0'], ja['right_j1'], ja['right_j2'], ja['right_j3'], ja['right_j4'], ja['right_j5'], ja['right_j6']])
# print(ja)
print(arm.joint_angles())
# import ipdb; ipdb.set_trace()