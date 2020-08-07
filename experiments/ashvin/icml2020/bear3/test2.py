import copy

import gym
import numpy as np
import torch.nn as nn

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.launchers.launcher_util import run_experiment
from rlkit.samplers.data_collector import MdpPathCollector # , CustomMdpPathCollector
from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.torch.networks import MlpQf, TanhMlpPolicy
from rlkit.torch.sac.policies import (
    TanhGaussianPolicy, VAEPolicy
)
from rlkit.torch.sac.bear import BEARTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rlkit.launchers.arglauncher import run_variants

import gym

DEFAULT_BUFFER = ('/media/avi/data/Work/doodad_output/20-03-31-railrl-bullet-'
                  'SawyerReach-v0-state/20-03-31-railrl-bullet-SawyerReach-v0-state'
                  '_2020_03_31_15_37_01_id123824--s443957/buffers/epoch_450.pkl')


def experiment(variant):
    import mj_envs

    expl_env = gym.make(variant['env'])
    eval_env = expl_env

    action_dim = int(np.prod(eval_env.action_space.shape))
    state_dim = obs_dim = np.prod(expl_env.observation_space.shape)
    M = 256

    qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
    qf_kwargs['output_size'] = 1
    qf_kwargs['input_size'] = action_dim + state_dim
    qf1 = MlpQf(**qf_kwargs)
    qf2 = MlpQf(**qf_kwargs)

    target_qf_kwargs = copy.deepcopy(qf_kwargs)
    target_qf1 = MlpQf(**target_qf_kwargs)
    target_qf2 = MlpQf(**target_qf_kwargs)

    policy_kwargs = copy.deepcopy(variant['policy_kwargs'])
    policy_kwargs['action_dim'] = action_dim
    policy_kwargs['obs_dim'] = state_dim
    policy = TanhGaussianPolicy(**policy_kwargs)

    vae_policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        latent_dim=action_dim * 2,
    )

    eval_path_collector = MdpPathCollector(
        eval_env,
        # save_images=False,
        vae_policy,
        **variant['eval_path_collector_kwargs']
    )

    vae_eval_path_collector = MdpPathCollector(
        eval_env,
        vae_policy,
        # max_num_epoch_paths_saved=5,
        # save_images=False,
    )


    replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )
    demo_train_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )
    demo_test_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )


    trainer = BEARTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vae=vae_policy,
        replay_buffer=replay_buffer,
        **variant['trainer_kwargs']
    )

    path_loader_class = variant.get('path_loader_class', MDPPathLoader)
    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    path_loader = path_loader_class(trainer,
                                    replay_buffer=replay_buffer,
                                    demo_train_buffer=demo_train_replay_buffer,
                                    demo_test_buffer=demo_test_replay_buffer,
                                    **path_loader_kwargs,
                                    # demo_off_policy_path=variant['data_path'],
                                    )
    # path_loader.load_bear_demos(pickled=False)
    path_loader.load_demos()
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        **variant['expl_path_collector_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        # vae_evaluation_data_collector=vae_eval_path_collector,
        replay_buffer=replay_buffer,
        # q_learning_alg=True,
        # batch_rl=variant['batch_rl'],
        **variant['algo_kwargs']
    )


    algorithm.to(ptu.device)
    trainer.pretrain_q_with_bc_data(256)
    algorithm.train()



if __name__ == "__main__":
    variant = dict(
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            target_update_method='default',

            # use_automatic_entropy_tuning=True,
            # BEAR specific params
            mode='auto',
            kernel_choice='laplacian',
            policy_update_style='0',  # 0 is min, 1 is average (for double Qs)
            mmd_sigma=20.0,  # 5, 10, 40, 50

            target_mmd_thresh=0.05,  # .1, .07, 0.01, 0.02

            # gradient penalty hparams
            with_grad_penalty_v1=False,
            with_grad_penalty_v2=False,
            grad_coefficient_policy=0.001,
            grad_coefficient_q=1E-4,
            start_epoch_grad_penalty=24000,

            # Target nets/ policy vs Q-function update
            use_target_nets=True,
            policy_update_delay=1,
            num_steps_policy_update_only=1,

            ## advantage weighting
            use_adv_weighting=False,
            pretraining_env_logging_period=10000,
            pretraining_logging_period=1000,
            do_pretrain_rollouts=False,
            num_pretrain_steps=25000,
        ),
        algo_kwargs=dict(
            batch_size=256,
            max_path_length=1000,
            num_epochs=1001,
            num_eval_steps_per_epoch=3000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            # min_num_steps_before_training=10*1000,
            # max_path_length=10,
            # num_epochs=100,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
        ),
        # data_path='demos/final_buffers_buffer_medium_cheetah.npz',
        # data_path=[
        #     'demos/ant_action_noise_15.npy',
        #     'demos/ant_off_policy_15_demos_100.npy',
        # ],
        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            obs_key="state_observation",
            demo_paths=[
                dict(
                    path="demos/icml2020/hand/pen2_sparse.npy",
                    obs_dict=True,
                    is_demo=True,
                ),
                dict(
                    path="demos/icml2020/hand/pen_bc_sparse4.npy",
                    obs_dict=False,
                    is_demo=False,
                    train_split=0.9,
                ),
            ],
        ),
        replay_buffer_size=int(1E6),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        expl_path_collector_kwargs=dict(render=False),
        eval_path_collector_kwargs=dict(render=False),
        shared_qf_conv=False,
        use_robot_state=True,
        batch_rl=False,
        # path_loader_kwargs=dict(
            # demo_paths=[
                # dict(
                    # path='demos/hc_action_noise_15.npy',
                    # obs_dict=False,
                    # is_demo=False,
                    # train_split=.9,
                # ),
                # dict(
                    # path='demos/hc_off_policy_15_demos_100.npy',
                    # obs_dict=False,
                    # is_demo=False,
                # ),
            # ],
        # ),

    )

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, default='Ant-v2',
    #                     choices=('SawyerReach-v0', 'SawyerGraspOne-v0'))
    # parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    # args = parser.parse_args()

    # variant['env'] = args.env
    # variant['obs'] = 'state'
    # variant['buffer'] = args.buffer

    n_seeds = 3
    # mode = 'local_docker'
    mode = 'local'
    # exp_prefix = 'dev-{}'.format(
    #     __file__.replace('/', '-').replace('_', '-').split('.')[0]
    # )
    exp_prefix = 'bear_ant_our_data_online_v1'

    # n_seeds = 5
    # mode = 'ec2'

    search_space = {
        'env': ['pen-binary-v0', ],
        'shared_qf_conv': [
            True,
            # False,
        ],
        'trainer_kwargs.kernel_choice':['laplacian'],
        # 'trainer_kwargs.target_mmd_thresh':[.15, .1, .07, .05, 0.01, 0.02],
        'trainer_kwargs.target_mmd_thresh':[.05, 0.02],
        'trainer_kwargs.num_samples_mmd_match':[4, 10],
        'trainer_kwargs.mmd_sigma':[50, ],
        'seedid': range(3),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, )

    # for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
    #     for _ in range(n_seeds):
    #         run_experiment(
    #             experiment,
    #             exp_name=exp_prefix,
    #             mode=mode,
    #             variant=variant,
    #             use_gpu=False,
    #             gpu_id=0,
    #             unpack_variant=False,
    #         )
