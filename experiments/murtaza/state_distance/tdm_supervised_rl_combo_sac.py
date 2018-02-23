"""
Example of running PyTorch implementation of DDPG on HalfCheetah.
"""
import gym

from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.reacher_7dof import Reacher7DofXyzGoalState
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.state_distance.tdm_sac import TdmSac
import railrl.torch.pytorch_util as ptu
from railrl.state_distance.tdm_networks import TdmQf, TdmVf, StochasticTdmPolicy
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
import numpy as np

def experiment(variant):
    env = NormalizedBoxEnv(Reacher7DofXyzGoalState())
    vectorized=True
    policy = StochasticTdmPolicy(
        env=env,
        **variant['policy_kwargs']
    )
    qf = TdmQf(
        env=env,
        vectorized=vectorized,
        norm_order=2,
        **variant['qf_kwargs']
    )
    vf = TdmVf(
        env=env,
        vectorized=vectorized,
        **variant['vf_kwargs']
    )
    replay_buffer_size = variant['algo_params']['base_kwargs']['replay_buffer_size']
    replay_buffer = HerReplayBuffer(replay_buffer_size, env)
    algorithm = TdmSac(
        env,
        qf,
        vf,
        variant['algo_params']['sac_kwargs'],
        variant['algo_params']['tdm_kwargs'],
        variant['algo_params']['base_kwargs'],
        policy=policy,
        replay_buffer=replay_buffer,
    )
    if ptu.gpu_enabled():
        algorithm.cuda()

    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        qf_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=500,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=99,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
                replay_buffer_size=1000000,
            ),
            tdm_kwargs=dict(
                max_tau=10,
            ),
            sac_kwargs=dict(
                soft_target_tau=0.01,
                policy_lr=3E-4,
                qf_lr=3E-4,
                vf_lr=3E-4,
            ),
            supervised_weight=.5,
        ),
    )
    search_space = {
        'algo_params.base_kwargs.reward_scale': [
            1,
            10,
            100,
        ],
        'algo_params.base_kwargs.num_updates_per_env_step': [
            1,
            10,
            20,
        ],
        'algo_params.tdm_kwargs.max_tau': [
            10,
            15,
            20,
        ],
        'algo_params.supervised_weight':[
            .1,
            .2,
            .3,
            .4,
            .5,
            .6,
            .7,
            .8,
            .9
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        run_experiment(
            experiment,
            seed=np.random.randint(1, 10004),
            variant=variant,
            exp_id=exp_id,
            exp_prefix='tdm_supervised_rl_combo_hyper_parameter_sweep',
            mode='local_docker',
        )
