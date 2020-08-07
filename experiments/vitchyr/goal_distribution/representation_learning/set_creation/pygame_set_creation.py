from rlkit.torch.sets import set_creation
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.envs.images import EnvRenderer


def main():
    env = PickAndPlaceEnv(
        # Environment dynamics
        action_scale=1.0,
        boundary_dist=4,
        ball_radius=1.5,
        object_radius=1.,
        cursor_visual_radius=1.5,
        object_visual_radius=1.,
        min_grab_distance=1.,
        walls=None,
        # Rewards
        action_l2norm_penalty=0,
        reward_type="dense",
        success_threshold=0.60,
        # Reset settings
        fixed_goal=None,
        # Visualization settings
        images_are_rgb=True,
        render_dt_msec=0,
        render_onscreen=False,
        render_size=84,
        show_goal=False,
        goal_samplers=None,
        goal_sampling_mode='random',
        num_presampled_goals=10000,
        object_reward_only=False,

        init_position_strategy='random',
        num_objects=2,
    )

    renderer = EnvRenderer(
        output_image_format='CHW',
    )
    sets = set_creation.sample_pnp_sets(
        env,
        renderer,
        num_samples_per_set=128,
        set_configs=[
            dict(
                version='project_onto_axis',
                axis_idx_to_value={
                    0: None,
                    1: None,
                },
            ),
            dict(
                version='project_onto_axis',
                axis_idx_to_value={
                    0: None,
                },
            ),
            dict(
                version='project_onto_axis',
                axis_idx_to_value={
                    2: None,
                    3: None,
                },
            ),
            dict(
                version='project_onto_axis',
                axis_idx_to_value={
                    2: None,
                },
            ),
        ],
    )
    set_creation.save(sets, 'hand2xy_hand2x_1obj2xy_1obj2x.pickle')


if __name__ == '__main__':
    main()