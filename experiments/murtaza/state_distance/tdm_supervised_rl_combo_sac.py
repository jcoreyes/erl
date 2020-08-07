"""
Example of running PyTorch implementation of DDPG on HalfCheetah.
"""
import gym

from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.envs.multitask.point2d import MultitaskPoint2DEnv
from rlkit.envs.multitask.reacher_7dof import Reacher7DofXyzGoalState
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.state_distance.tdm_sac import TdmSac
import rlkit.torch.pytorch_util as ptu
from rlkit.state_distance.tdm_networks import TdmQf, TdmVf, StochasticTdmPolicy
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
import numpy as np

def experiment(variant):
    # env = NormalizedBoxEnv(Reacher7DofXyzGoalState())
    env = NormalizedBoxEnv(MultitaskPoint2DEnv())
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
        supervised_weight=variant['algo_params']['supervised_weight'],
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
                num_epochs=100,
                num_steps_per_epoch=100,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
                reward_scale=100,
                replay_buffer_size=1000000,
                render=True,
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
        ),
    )
    search_space = {
        'algo_params.base_kwargs.reward_scale': [
            1,
            # 10,
            # 100,
        ],
        'algo_params.tdm_kwargs.max_tau': [
            1,
            # 15,
            # 20,
        ],
        'algo_params.supervised_weight':[
             # 0,
            .2,
            # .4,
            # .6,
            # .8,
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
            # exp_prefix='tdm_rl_supervised_combo',
            exp_prefix='tdm_rl_supervised_combo',
            mode='local',
        )
