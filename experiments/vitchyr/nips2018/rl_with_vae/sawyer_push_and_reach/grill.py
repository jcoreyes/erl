from railrl.envs.mujoco.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYEasyEnv
from railrl.envs.mujoco.sawyer_push_env import SawyerPushXYEnv
from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
from railrl.envs.multitask.pusher2d import FullPusher2DEnv
from railrl.images.camera import sawyer_init_camera, \
    sawyer_init_camera_zoomed_in

from railrl.launchers.arglauncher import run_variants
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'grill-sawyer-push-oracle-vae-with-variable-training-longer'

    # zoomed_in_path = "05-22-vae-sawyer-new-push-easy-zoomed-in-1000_2018_05_22_13_09_28_0000--s-98682-r16/params.pkl"
    # zoomed_out_path = "05-22-vae-sawyer-new-push-easy-no-zoom-1000_2018_05_22_13_10_43_0000--s-30039-r16/params.pkl"
    zoomed_in_path = "05-22-vae-sawyer-variable-zoomed-in/05-22-vae-sawyer" \
                     "-variable-zoomed-in_2018_05_22_20_56_11_0000--s-10690" \
                     "-r16/params.pkl"
    zoomed_out_path = "05-22-vae-sawyer-variable-no-zoom/05-22-vae-sawyer" \
                      "-variable-no-zoom_2018_05_22_20_59_07_0000--s-40296" \
                      "-r16/params.pkl"

    vae_paths = {
        # "4": "05-12-vae-sawyer-new-push-easy-3/05-12-vae-sawyer-new-push-easy"
        #       "-3_2018_05_12_02_00_01_0000--s-91524-r4/params.pkl",
        # "16": "05-12-vae-sawyer-new-push-easy-3/05-12-vae-sawyer-new-push"
        #       "-easy-3_2018_05_12_02_33_54_0000--s-1937-r16/params.pkl",
        # "16b": zoomed_in_path,
        # "64": "05-12-vae-sawyer-new-push-easy-3/05-12-vae-sawyer-new-push"
        #       "-easy-3_2018_05_12_03_06_20_0000--s-33176-r64/params.pkl",
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=500,
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
        init_camera=sawyer_init_camera_zoomed_in,
    )

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.num_updates_per_env_step': [1],
        'exploration_noise': [0.2],
        'algo_kwargs.reward_scale': [1e-4],
        'reward_params.type': [
            # 'mahalanobis_distance',
            # 'log_prob',
            'latent_distance',
        ],
        'training_mode': ['train'],
        'testing_mode': ['test', ],
        # 'rdim': ['16b', '4', '16', '64'],
        'rdim': ['16'],
        'init_camera': [
            sawyer_init_camera,
            sawyer_init_camera_zoomed_in,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if variant['init_camera'] == sawyer_init_camera_zoomed_in:
            variant['vae_paths']['16'] = zoomed_in_path
        elif variant['init_camera'] == sawyer_init_camera:
            variant['vae_paths']['16'] = zoomed_out_path
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                # snapshot_gap=50,
                # snapshot_mode='gap_and_last',
            )
