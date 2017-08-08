from railrl.envs.ros.sawyer_env import SawyerEnv
import numpy as np

def create_action(mag):
    action = np.random.rand((1, 7))[0]


experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]

env = SawyerEnv('right', experiment=experiments[0], safety_force_magnitude=5)

env.reset()
ja = env._joint_angles()
# ja =np.array([-2.13281250e-03, -1.18177441e+00, -2.75390625e-03, 2.17755176e+00,
#          2.20019531e-03, 5.67653320e-01, 3.31843457e+00])
# ja = env._wrap_angles(ja)
# ja = np.array([ja['right_j0'], ja['right_j1'], ja['right_j2'], ja['right_j3'], ja['right_j4'], ja['right_j5'], ja['right_j6']])
print(ja)
# env.arm.move_to_neutral()
#
# for _ in range(10):
#     action = create_action(.1)
#     env.step(action)

# while True:
#     # env.safety_box_check(reset_on_error=False)
#     # env.high_torque_check()
#     # env._act(-5*np.ones(7))
#     # env.unexpected_torque_check(reset_on_error=False)
#     if (env.get_observed_torques_minus_gravity() > 5).any():
#         print(env.get_observed_torques_minus_gravity())
# angles = env.arm.joint_angles()
# angles = np.array([angles['right_j0'], angles['right_j1'], angles['right_j2'], angles['right_j3'], angles['right_j4'], angles['right_j5'], angles['right_j6']])

# print(env._wrap_angles(angles))