import intera_interface as ii
import rospy
rospy.init_node('right')
arm = ii.Limb('right')
arm.move_to_neutral()
# pos = arm.endpoint_pose()['position']
# print([pos.x, pos.y, pos.z])