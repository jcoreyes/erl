from railrl.envs.multitask.sawyer_env import MultiTaskSawyerEnv
from railrl.envs.ros.sawyer_env import SawyerEnv
import numpy as np
import intera_interface
from intera_interface import CHECK_VERSION

def create_action(mag):
    action = mag*np.random.rand(1, 7)[0] - mag / 2
    return action

experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]

env = SawyerEnv('right', experiment=experiments[4], loss='huber', safety_force_magnitude=5, temp=15, safety_box=True, use_safety_checks=False)
# env.reset()
# for i in range(100000):
#     joint_to_values = dict(zip(env.arm_joint_names, np.zeros(7)))
#     env.arm.set_joint_torques(joint_to_values)
#     env.rate.sleep()
# env.arm.move_to_neutral()
# ja = env._joint_angles()
# # ja =np.array([-2.13281250e-03, -1.18177441e+00, -2.75390625e-03, 2.17755176e+00,
# #          2.20019531e-03, 5.67653320e-01, 3.31843457e+00])
# # ja = env._wrap_angles(ja)
# # ja = np.array([ja['right_j0'], ja['right_j1'], ja['right_j2'], ja['right_j3'], ja['right_j4'], ja['right_j5'], ja['right_j6']])
# print(ja)
# env.arm.move_to_neutral()
#
# for _ in range(10):
#     action = create_action(.1)
#     env.step(action)
# try:
#     for _ in range(1000):
#         print(1)
# except Exception as e:
#     import ipdb; ipdb.set_trace()
# print(env.is_in_box(env.arm.endpoint_pose))
# # while True:
pose = env.arm.endpoint_pose()['position']
orientation = env.arm.endpoint_pose()['orientation']
while True:
    print(env.rewards(np.zeros(7)))
    # env.rewards(np.zeros(7))
pose = np.array([pose.x, pose.y, pose.z, orientation.x, orientation.y, orientation.z, orientation.w])
# pose = np.array([pose.x, pose.y, pose.z])
print('[', end='')
for thing in pose:
    print(str(thing), end=', ')
print(']')
# env.update_pose_and_jacobian_dict()
# env.check_joints_in_box(env.pose_jacobian_dict)
# print(env.pose_jacobian_dict)
# #
# des = np.array([0.68998028, -0.2285752, 0.3477])
#
# print(np.linalg.norm(pose-des))
# for i in range(100000):
#     env._act(np.zeros(7))
#     action = create_action(2)
#     env._act(action)
#     if i % 200 == 0:
#         env.reset()
#     env.update_n_step_buffer(env._get_observation(), np.zeros(7), 0)
    # env.safety_box_check(reset_on_error=False)
    # env.unexpected_velocity_check(reset_on_error=True)
    # env.high_torque_check()
    # print('je;;p')
    # print('gravity subtracted torques: ', env.get_observed_torques_minus_gravity())
    # env.unexpected_torque_check(reset_on_error=False)
    # env.unexpected_velocity_check(reset_on_error=False)
#     # env.safety_box_check(reset_on_error=False)
#     # env.high_torque_check()
#     # env._act(-5*np.ones(7))
#     env.unexpected_torque_check(reset_on_error=False)
#     if (env.get_observed_torques_minus_gravity() > 5).any():
#         print(env.get_observed_torques_minus_gravity())
# np.save('buffer_with_pushes.npy', env.q)
# angles = env.arm.joint_angles()
# angles = np.array([angles['right_j0'], angles['right_j1'], angles['right_j2'], angles['right_j3'], angles['right_j4'], angles['right_j5'], angles['right_j6']])

# print(env._wrap_angles(angles))
#
# import rospy
# rospy.init_node('saw')
# rs = intera_interface.RobotEnable(CHECK_VERSION)
# init_state = rs.state().enabled
# import ipdb; ipdb.set_trace(    )
# rs.state()

#back left: [-0.04304189 -0.43462352  0.16761519]

#front right: [ 0.84045825  0.38408276  0.8880568 ]