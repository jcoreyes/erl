env_kwargs={
    'action_scale': .06,
    'action_repeat': 10,
    'timestep': 1./120,
    'solver_iterations': 500,
    'max_force': 1000,

    'gui': True,
    'pos_init': [.75, -.3, 0],
    'pos_high': [.75, .4, .3],
    'pos_low': [.75, -.4, -.36],
    'reset_obj_in_hand_rate': 0.0,
    'goal_sampling_mode': 'ground',
    'random_init_bowl_pos': True,
    'bowl_type': 'fixed',
    'bowl_bounds': [-0.40, 0.40],

    'hand_reward': True,
    'gripper_reward': True,
    'bowl_reward': True,

    'use_rotated_gripper': True,
    'use_wide_gripper': True,
    'soft_clip': True,
    'obj_urdf': 'spam',
    'max_joint_velocity': None,
}