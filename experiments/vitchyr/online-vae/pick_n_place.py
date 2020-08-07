import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
    import SawyerPickAndPlaceEnvYZ
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment

if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'grill-pick-and-place-yz'
    vae_path = "06-26-multiworld-pick-vae/06-26-multiworld-pick-vae_2018_06_26_23_20_15_0000--s-92887/vae.pkl"
#    vae_path = "06-27-multiworld-pick-vae/06-27-multiworld-pick-vae_2018_06_27_17_14_41_0000--s-28626/vae_T.pkl"
    # Multi Objects
#    vae_path = "06-28-multiworld-pick-vae/06-28-multiworld-pick-vae_2018_06_28_11_18_18_0000--s-64395/vae.pkl"
    # smaller goal space
    vae_path = "06-29-multiworld-pick-vae-smaller-goal-space/06-29-multiworld-pick-vae-smaller-goal-space_2018_06_29_00_55_51_0000--s-96350/vae.pkl"
    # smaller goal space .3Z
    #vae_path = "06-29-multiworld-pick-vae-smaller-goal-space/06-29-multiworld-pick-vae-smaller-goal-space_2018_06_29_12_44_30_0000--s-75713/vae.pkl"
    # small block
    # vae_path = "06-29-multiworld-pick-vae-smaller-goal-space-small-block/06-29-multiworld-pick-vae-smaller-goal-space-small-block_2018_06_29_14_09_16_0000--s-35469/vae.pkl"
    # confirm .2
#    vae_path = "06-29-multiworld-pick-vae-smaller-goal-space-reproduce/06-29-multiworld-pick-vae-smaller-goal-space-reproduce_2018_06_29_16_44_12_0000--s-89480/vae.pkl"

    # Visible arm smaller goal range
    vae_path = "06-30-multiworld-pick-vae-smaller-goal-space-reproduce/06-30-multiworld-pick-vae-smaller-goal-space-reproduce_2018_06_30_18_33_09_0000--s-47869/vae.pkl"

    variant = dict(
        env_kwargs=dict(
            hide_arm=False,
            hide_goal_markers=True,
            # random_init=True,
        ),

        env_class=SawyerPickAndPlaceEnvYZ,
        init_camera=sawyer_pick_and_place_camera,
        train_vae_variant = dict(
            beta=5.0,
            representation_size=8,
            generate_vae_dataset_kwargs=dict(
                N=10000,
                use_cached=False,
                imsize=84,
                num_channels=3,
                show=False,
                oracle_dataset=True,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
            ),
            beta_schedule_kwargs=dict(
                x_values=[0, 100, 200, 500, 1000],
                y_values=[0, 0, 0, 2.5, 5],
            ),
            save_period=5,
            num_epochs=1000,
        ),
        grill_variant = dict(
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=1000,
                    num_steps_per_epoch=500,
                    num_steps_per_eval=500,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    min_num_steps_before_training=1000,
                    collection_mode='online-parallel',
                ),
                her_kwargs=dict(),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
            ),
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            replay_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='HER-TD3',
            # vae_path="08-06-grill-pick-and-place-yz/08-06-grill-pick-and-place-yz_2018_08_06_12_54_01_0000--s-55708/vae.pkl",
            normalize=False,
            render=False,
            use_env_goals=True,
            wrap_mujoco_env=True,
            do_state_based_exp=True,
            exploration_noise=0.3,
        )
    )

    search_space = {
        'grill_variant.exploration_type': [
            'ou'
        ],
        'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [.5],
        'train_vae_variant.representation_size': [6],
        'grill_variant.algo_kwargs.num_updates_per_env_step': [1],
        'env_kwargs.oracle_reset_prob': [0.0],
        'env_kwargs.random_init': [False],
        'env_kwargs.action_scale': [.02],
        'grill_variant.exploration_noise': [.5],
        'grill_variant.algo_kwargs.reward_scale': [1],
        'grill_variant.reward_params.type': [
            'latent_distance',
        ],
        'grill_variant.training_mode': ['train'],
        'grill_variant.testing_mode': ['test', ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                snapshot_gap=100,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=3,
            )