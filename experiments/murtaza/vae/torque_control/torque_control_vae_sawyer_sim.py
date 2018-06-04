from railrl.envs.mujoco.sawyer_reach_torque_env import SawyerReachTorqueEnv
from railrl.images.camera import sawyer_torque_env_camera
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    n_seeds = 1
    mode = 'local_docker'
    exp_prefix = 'sawyer_torque_control_ou_vae'

    vae_paths = {
     # "16": "/home/murtaza/Documents/rllab/railrl/experiments/murtaza/vae/torque_control/torque_params.pkl",
     "16": "/home/murtaza/Documents/rllab/railrl/experiments/murtaza/vae/torque_control/ou_torque_params.pkl",
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=500,
            num_steps_per_epoch=50,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=50,
            discount=0.99,
        ),
        env_kwargs=dict(
            hide_goal=True,
        ),
        replay_kwargs=dict(
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        algorithm='HER-TD3',
        normalize=True,
        rdim=16,
        render=False,
        env=SawyerReachTorqueEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        do_state_based_exp=False,
        exploration_noise=0.1,
        init_camera=sawyer_torque_env_camera,
        version='normal',
        reward_params=dict(
        ),
        use_gpu=True
    )

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.num_updates_per_env_step': [4],
        'replay_kwargs.fraction_goals_are_env_goals': [0.5, 1.0],
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2],
        'exploration_noise': [0.2],
        'algo_kwargs.reward_scale': [1e-4,],
        'training_mode': ['train'],
        'testing_mode': ['test', ],
        'rdim': [16],
        'reward_params.type': ['latent_distance', 'log_prob'],
        'history_len':[2],
        'hidden_sizes':[[300, 400, 300], [400, 300]]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if (
                variant['replay_kwargs']['fraction_goals_are_rollout_goals'] == 1.0
                and variant['replay_kwargs']['fraction_goals_are_env_goals'] == 0.5
        ):
            # redundant setting
            continue
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
            )