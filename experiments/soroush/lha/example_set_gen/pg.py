from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.sets.example_set_gen import gen_example_sets_full_experiment
from multiworld.envs.pygame.pick_and_place import PickAndPlaceEnv

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
    env_class=PickAndPlaceEnv,
    env_kwargs=dict(
        # Environment dynamics
        action_scale=1.0,
        ball_radius=1.0,  # 1.
        boundary_dist=4,
        object_radius=1.0,
        min_grab_distance=1.0,
        walls=None,
        # Rewards
        action_l2norm_penalty=0,
        reward_type="dense",  # dense_l1
        object_reward_only=False,
        success_threshold=0.60,
        # Reset settings
        fixed_goal=None,
        # Visualization settings
        images_are_rgb=True,
        render_dt_msec=0,
        render_onscreen=False,
        render_size=84,
        show_goal=False,  # True
        # get_image_base_render_size=(48, 48),
        # Goal sampling
        goal_samplers=None,
        goal_sampling_mode='random',
        num_presampled_goals=10000,

        init_position_strategy='random',

        num_objects=4,
    ),
    imsize=256,
)

if __name__ == "__main__":
    run_experiment(
        method_call=gen_example_sets_full_experiment,
        variant=variant,
        exp_prefix='pg-example-set', # change to exp_name is this doesn't work
    )
