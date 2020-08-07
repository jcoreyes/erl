from rlkit.envs.mujoco.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYEasyEnv
from rlkit.envs.mujoco.sawyer_push_env import SawyerPushXYEnv
from rlkit.envs.multitask.point2d import MultitaskImagePoint2DEnv
from rlkit.envs.multitask.pusher2d import FullPusher2DEnv
from rlkit.images.camera import (
    # sawyer_init_camera,
    # sawyer_init_camera_zoomed_in,
    sawyer_init_camera_zoomed_in_fixed,
    sawyer_init_camera_zoomed_out_fixed,
)

from rlkit.launchers.arglauncher import run_variants
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'grill-her-td3-sawyer-push-zoomed-in-fixed-softtau-sweep-2'

    # zoomed_in_path = "05-22-vae-sawyer-new-push-easy-zoomed-in-1000_2018_05_22_13_09_28_0000--s-98682-r16/params.pkl"
    # zoomed_out_path = "05-22-vae-sawyer-new-push-easy-no-zoom-1000_2018_05_22_13_10_43_0000--s-30039-r16/params.pkl"
    zoomed_in_path = "05-22-vae-sawyer-variable-zoomed-in/05-22-vae-sawyer" \
                     "-variable-zoomed-in_2018_05_22_20_56_11_0000--s-10690" \
                     "-r16/params.pkl"
    # zoomed_out_path = (
        # "05-22-vae-sawyer-variable-no-zoom_2018_05_22_21_46_54_0000--s-76844"
        # "-r16params.pkl"
    # )
    zoomed_out_path = (
        "05-22-vae-sawyer-variable-no-zoom-300-epochs_2018_05_22_21_55_41_0000--s-48052-r16"
        "/params.pkl"
    )

    vae_paths = {
        # "4": "05-12-vae-sawyer-new-push-easy-3/05-12-vae-sawyer-new-push-easy"
        #       "-3_2018_05_12_02_00_01_0000--s-91524-r4/params.pkl",
        # "16": "05-12-vae-sawyer-new-push-easy-3/05-12-vae-sawyer-new-push"
              # "-easy-3_2018_05_12_02_33_54_0000--s-1937-r16/params.pkl",
        # "16": "05-23-vae-sawyer-variable-zoomed-out-sweep/05-23-vae-sawyer-variable-zoomed-out-sweep-id0-s31952-nImg-1000--cam-sawyer_init_camera_zoomed_out_fixed/params.pkl",
        # "16": "05-23-vae-sawyer-variable-zoomed-out-sweep/05-23-vae-sawyer-variable-zoomed-out-sweep-id0-s52951-nImg-1000--cam-sawyer_init_camera_zoomed_out_fixed/params.pkl",
        "16": "05-23-vae-sawyer-variable-fixed-2/05-23-vae-sawyer-variable"
              "-fixed-2_2018_05_23_16_19_33_0000--s-293-nImg-1000--cam-sawyer_init_camera_zoomed_in_fixed/params.pkl",
        # "16b": zoomed_in_path,
        # "64": "05-12-vae-sawyer-new-push-easy-3/05-12-vae-sawyer-new-push"
        #       "-easy-3_2018_05_12_03_06_20_0000--s-33176-r64/params.pkl",
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=250,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            min_num_steps_before_training=1000,
        ),
        env_kwargs=dict(
            hide_goal=True,
            # reward_info=dict(
            #     type="shaped",
            # ),
        ),
        replay_kwargs=dict(
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        algorithm='HER-TD3',
        normalize=False,
        rdim=4,
        render=False,
        env=SawyerPushAndReachXYEasyEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        do_state_based_exp=False,
        exploration_noise=0.1,
        init_camera=sawyer_init_camera_zoomed_in_fixed,
    )

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.num_updates_per_env_step': [1, 4, 8],
        'algo_kwargs.tau': [1, 1e-1, 1e-2, 1e-3],
        'exploration_noise': [0.2],
        'algo_kwargs.reward_scale': [1],
        'reward_params.type': [
            # 'mahalanobis_distance',
            # 'log_prob',
            'latent_distance',
        ],
        'training_mode': ['train'],
        'testing_mode': ['test', ],
        # 'rdim': ['16b', '4', '16', '64'],
        'rdim': ['16'],
        # 'init_camera': [
            # # sawyer_init_camera,
            # sawyer_init_camera_zoomed_out_fixed,
        # ],
        # 'vae_paths.16': [
            # '05-23-vae-sawyer-variable-fixed-2/'
            # '05-23-vae-sawyer-variable-fixed-2_2018_05_23_16_19_02_0000--s-27304-nImg-100--cam-sawyer_init_camera_zoomed_in_fixed/params.pkl',
            # '05-23-vae-sawyer-variable-fixed-2/'
            # '05-23-vae-sawyer-variable-fixed-2_2018_05_23_16_19_17_0000--s-62873-nImg-100--cam-sawyer_init_camera_zoomed_out_fixed/params.pkl',
            # '05-23-vae-sawyer-variable-fixed-2/'
            # '05-23-vae-sawyer-variable-fixed-2_2018_05_23_16_19_26_0000--s-50588-nImg-1000--cam-sawyer_init_camera_zoomed_out_fixed/params.pkl',
            # '05-23-vae-sawyer-variable-fixed-2/'
            # '05-23-vae-sawyer-variable-fixed-2_2018_05_23_16_19_33_0000--s-293-nImg-1000--cam-sawyer_init_camera_zoomed_in_fixed/params.pkl',
        # ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        # if variant['init_camera'] == sawyer_init_camera_zoomed_in:
        #     variant['vae_paths']['16'] = zoomed_in_path
        # elif variant['init_camera'] == sawyer_init_camera:
        #     variant['vae_paths']['16'] = zoomed_out_path
        # zoomed = 'zoomed_out' not in variant['vae_paths']['16']
        # n1000 = 'nImg-1000' in variant['vae_paths']['16']
        # if zoomed:
            # variant['init_camera'] = sawyer_init_camera_zoomed_out_fixed
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                # trial_dir_suffix='n1000-{}--zoomed-{}'.format(n1000, zoomed),
                snapshot_gap=50,
                snapshot_mode='gap_and_last',
            )
