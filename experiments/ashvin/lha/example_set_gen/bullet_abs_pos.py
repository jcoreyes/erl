from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.sets.example_set_gen import gen_example_sets_full_experiment
from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC

variant = dict(
do_state_exp=True,
    example_set_variant=dict(
        n=30,
        subtask_codes=[
            {2: 2, 3: 3},
            {4: 4, 5: 5},
            {6: 6, 7: 7},
            {8: 8, 9: 9},
        ],
        other_dims_random=True,
    ),
    env_class=SawyerLiftEnvGC,
    env_kwargs={
        'action_scale': .06,
        'action_repeat': 10,
        'timestep': 1./120,
        'solver_iterations': 500,
        'max_force': 1000,

        'gui': False,
        'pos_init': [.75, -.3, 0],
        'pos_high': [.75, .4, .3],
        'pos_low': [.75, -.4, -.36],
        'reset_obj_in_hand_rate': 0.0,
        'bowl_bounds': [-0.40, 0.40],

        'use_rotated_gripper': True,
        'use_wide_gripper': True,
        'soft_clip': True,
        'obj_urdf': 'spam',
        'max_joint_velocity': None,

        'hand_reward': True,
        'gripper_reward': True,
        'bowl_reward': True,

        'goal_sampling_mode': 'ground',
        'random_init_bowl_pos': False,
        'bowl_type': 'fixed',

        'num_obj': 4,
    },
    imsize=400,
)

if __name__ == "__main__":
    run_experiment(
        method_call=gen_example_sets_full_experiment,
        variant=variant,
        exp_prefix='pb-abs-example-set', # change to exp_name is this doesn't work
        mode="here_no_doodad",
    )
