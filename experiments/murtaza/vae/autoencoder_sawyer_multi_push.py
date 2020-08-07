from rlkit.envs.mujoco.sawyer_push_and_reach_env import SawyerMultiPushAndReachEasyEnv
from rlkit.images.camera import sawyer_init_camera_zoomed_in
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    n_seeds = 5
    mode = 'ec2'
    exp_prefix = 'sawyer_multi_push_autoencoder_ablation_final'

    vae_paths = {
         # # "16": "/home/murtaza/Documents/rllab/rlkit/data/local/05-17-sawyer-multi-push-ae/05-17-sawyer_multi_push_ae_2018_05_17_12_37_03_0000--s-9903/itr_30.pkl"
         # "16": "05-17-sawyer-multi-push-ae/05-17-sawyer_multi_push_ae_2018_05_17_12_37_03_0000--s-9903/itr_30.pkl"
        "4": "/home/murtaza/Documents/rllab/railrl/experiments/murtaza/vae/multi_push.pkl"
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
            num_updates_per_env_step=4,
        ),
        env_kwargs=dict(
            hide_goal=True,
        ),
        replay_kwargs=dict(
            fraction_goals_are_rollout_goals=1,
            fraction_resampled_goals_are_env_goals=0,
        ),
        algorithm='HER-TD3',
        normalize=False,
        rdim=16,
        render=False,
        env=SawyerMultiPushAndReachEasyEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        do_state_based_exp=False,
        exploration_noise=0.1,
        init_camera=sawyer_init_camera_zoomed_in,
        version='normal',
        reward_params=dict(
            min_variance=0,
        ),
        use_gpu=True,
        history_len=1,
    )

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'exploration_noise': [0.2],
        'algo_kwargs.reward_scale': [1e-4],
        'training_mode': ['train_env_goals'],
        'testing_mode': ['test', ],
        'rdim': [4],
        'reward_params.type': ['latent_distance'],
        'vae_wrapped_env_kwargs.sample_from_true_prior': [False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
            )
