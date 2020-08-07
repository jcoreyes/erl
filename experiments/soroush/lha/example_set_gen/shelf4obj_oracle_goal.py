from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.sets.example_set_gen import gen_example_sets_full_experiment
from furniture.env.furniture_multiworld import FurnitureMultiworld

variant = dict(
do_state_exp=True,
    example_set_variant=dict(
        n=50,
        subtask_codes=[
            {17: 17, 18: 18, 19: 19},
            {14: 14, 15: 15, 16: 16},
            {11: 11, 12: 12, 13: 13},
        ],
        other_dims_random=True,
    ),
    env_class=FurnitureMultiworld,
    env_kwargs=dict(
        name="FurnitureCursorRLEnv",
        unity=False,
        tight_action_space=True,
        preempt_collisions=True,
        boundary=[0.5, 0.5, 0.95],
        pos_dist=0.2,
        num_connect_steps=0,
        num_connected_ob=False,
        num_connected_reward_scale=5.0,
        goal_type='zeros',  # reset
        reset_type='var_2dpos+no_rot',  # 'var_2dpos+var_1drot', 'var_2dpos+objs_near',

        control_degrees='3dpos+select+connect',
        obj_joint_type='slide',
        connector_ob_type=None,  # 'dist',

        move_speed=0.05,

        reward_type='state_distance',

        clip_action_on_collision=True,

        light_logging=True,

        furniture_name='shelf_ivar_0678_4obj_bb',
        anchor_objects=['1_column'],
        goal_sampling_mode='uniform',
        task_type='select2+move2',
    ),
    imsize=256,
)

if __name__ == "__main__":
    run_experiment(
        method_call=gen_example_sets_full_experiment,
        variant=variant,
        exp_prefix='shelf-4obj-oracle-goal-example-set', # change to exp_name is this doesn't work
    )
