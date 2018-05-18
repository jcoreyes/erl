from railrl.envs.mujoco.sawyer_push_env import SawyerPushXYEasyEnv
from railrl.images.camera import sawyer_init_camera_zoomed_in
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    n_seeds = 5
    mode = 'ec2'
    exp_prefix = 'sawyer_single_push_autoencoder_ablation_final'

    vae_paths = {
         # "16": "/home/murtaza/Documents/rllab/railrl/data/local/05-17-sawyer-push-easy-ae/05-17-sawyer_push_easy_ae_2018_05_17_12_00_00_0000--s-49686/iter_30.pkl"
     "16": "/home/murtaza/Documents/rllab/railrl/experiments/murtaza/vae/single_pusher.pkl"
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
            fraction_goals_are_env_goals=0,
        ),
        algorithm='HER-TD3',
        normalize=False,
        rdim=16,
        render=False,
        env=SawyerPushXYEasyEnv,
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
        'rdim': [16],
        'reward_params.type': ['latent_distance'],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        # if (
        #         variant['replay_kwargs']['fraction_goals_are_rollout_goals'] == 1.0
        #         and variant['replay_kwargs']['fraction_goals_are_env_goals'] == 0.5
        # ):
            # redundant setting
            # continue
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
            )
