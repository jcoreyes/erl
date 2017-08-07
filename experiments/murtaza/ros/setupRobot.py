import intera_interface as ii
import rospy
import numpy as np
rospy.init_node('right')
rate = rospy.Rate(20)
arm = ii.Limb('right')
arm.move_to_neutral()
# pos = arm.endpoint_pose()['position']
# print([pos.x, pos.y, pos.z])
for _ in range(100):
    action = np.ones(7)
    joint_to_values = dict(zip(arm.joint_names(), action))
    arm.set_joint_torques(joint_to_values)
    rate.sleep()

print(arm.joint_angles())
# import ipdb; ipdb.set_trace()