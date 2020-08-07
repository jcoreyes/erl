from rlkit.envs.mujoco.sawyer_gripper_env import SawyerXYEnv
from rlkit.images.camera import sawyer_init_camera
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    n_seeds = 5
    mode = 'ec2'
    exp_prefix = 'sawyer_reacher_autoencoder_ablation_final'

    vae_paths = {
         # "16": "/home/murtaza/Documents/rllab/rlkit/data/local/05-15-sawyer-reacher-ae/05-15-sawyer_reacher_ae_2018_05_15_15_07_14_0000--s-92932/params.pkl"
        "4": "/home/murtaza/Documents/rllab/railrl/experiments/murtaza/vae/reacher.pkl"
        # "4": "/home/murtaza/Documents/rllab/rlkit/data/local/05-25-sawyer-pos-reacher-ae-smaller-latents/05-25-sawyer_pos_reacher_ae_smaller_latents_2018_05_25_11_04_33_0000--s-3459/itr_90.pkl"
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=1000,
            num_steps_per_epoch=50,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=50,
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
        env=SawyerXYEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        do_state_based_exp=False,
        exploration_noise=0.1,
        init_camera=sawyer_init_camera,
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
        'algo_kwargs.num_updates_per_env_step': [4],
        'replay_kwargs.fraction_resampled_goals_are_env_goals': [0.0],
        'replay_kwargs.fraction_goals_are_rollout_goals': [1.0],
        'exploration_noise': [0.2],
        'algo_kwargs.reward_scale': [1e-4],
        'training_mode': ['test'],
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
