import baxter_interface as bi
import rospy

rospy.init_node("bax", anonymous=True)
arm = bi.Limb('left')
arm.move_to_neutral()
print(arm.joint_angles())