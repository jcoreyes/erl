#!/usr/bin/env python
import rospy
from intera_core_msgs.msg import SEAJointState



def callback(data):
    rospy.loginfo(data.actual_effort)
    # print(data.actual_effort)


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("robot/limb/right/gravity_compensation_torques", SEAJointState, callback)

    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()
    # for _ in range(10):
    #     continue
    rospy.wait_for_message("robot/limb/right/gravity_compensation_torques", SEAJointState)

if __name__ == '__main__':
    listener()
    # listener()