import os.path as osp
import multiworld.envs.mujoco as mwmj
from torch.nn import functional as F
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.experiments.disentanglement.state_launcher import \
    disentangled_her_twin_sac_experiment_v2

if __name__ == "__main__":
    variant = dict(
        # env_id='Point2DEnv-Train-Axis-Eval-Everything-v0',
        env_id='Point2DEnv-Train-Everything-Eval-Everything-v1',
        disentangled_qf_kwargs=dict(
            encode_state=False,
        ),
        qf_kwargs=dict(
            hidden_sizes=[64, 64],
            # hidden_sizes=[],
        ),
        policy_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        encoder_kwargs=dict(
            hidden_sizes=[64, 64],
            hidden_activation=F.tanh,
        ),
        twin_sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        max_path_length=100,
        algo_kwargs=dict(
            # batch_size=1024,
            # num_epochs=150,
            # num_eval_steps_per_epoch=2000,
            # num_expl_steps_per_train_loop=2000,
            # num_trains_per_train_loop=1000,
            # min_num_steps_before_training=1000,
            # batch_size=256,
            # num_epochs=20,
            # num_eval_steps_per_epoch=200,
            # num_expl_steps_per_train_loop=200,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
            batch_size=256,
            num_epochs=100,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        replay_buffer_kwargs=dict(
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
            max_size=int(1e6),
        ),
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        exploration_goal_sampling_mode='test',
        evaluation_goal_sampling_mode='test',
        save_video=False,
        save_video_kwargs=dict(
            save_video_period=10,
        ),
    )

    search_space = {
        'disentangled_qf_kwargs.give_each_qf_single_goal_dim': [
            True,
            False,
        ],
        'disentangled_qf_kwargs.encode_state': [
            True,
            # False,
        ],
        'exploration_goal_sampling_mode': ['train'],
        'encoder_kwargs.hidden_activation': [
            F.tanh,
            F.relu,
            F.leaky_relu,
        ],
        'latent_dim': [
            # 8,
            2, 4, 8
        ],
        # 'twin_sac_trainer_kwargs.qf_lr': [1e-4],
        'env_id': [
            # 'Point2DEnv-Train-Half-Axis-Eval-Everything-v0',
            # 'Point2DEnv-Train-Axis-Eval-Everything-v0',
            'Point2DEnv-Train-Everything-Eval-Everything-v1',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = '{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'net-64-64-sweep-encoder-activation-splice-latentdim-with-encode-state-2'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                disentangled_her_twin_sac_experiment_v2,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=False,
                num_exps_per_instance=3,
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=int(10*60),
              )
