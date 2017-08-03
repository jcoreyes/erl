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

env = SawyerEnv('right', experiment=experiments[0])

env.reset()
# env.arm.move_to_neutral()
#
# for _ in range(10):
#     action = create_action(.1)
#     env.step(action)

# while True:
    # env.safety_check()
    # print(env.get_torques())
angles = env.arm.joint_angles()
angles = np.array([angles['right_j0'], angles['right_j1'], angles['right_j2'], angles['right_j3'], angles['right_j4'], angles['right_j5'], angles['right_j6']])

print(env._wrap_angles(angles))