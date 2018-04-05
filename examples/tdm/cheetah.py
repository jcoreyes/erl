import torch.nn as nn

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.state_distance.tdm_ddpg import TdmDdpg
from railrl.state_distance.tdm_networks import TdmPolicy, \
    TdmQf, TdmNormalizer
from railrl.torch.modules import HuberLoss


def experiment(variant):
    env = NormalizedBoxEnv(GoalXVelHalfCheetah())
    max_tau = variant['ddpg_tdm_kwargs']['tdm_kwargs']['max_tau']
    tdm_normalizer = TdmNormalizer(
        env,
        vectorized=True,
        max_tau=max_tau,
    )
    qf = TdmQf(
        env=env,
        vectorized=True,
        tdm_normalizer=tdm_normalizer,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        tdm_normalizer=tdm_normalizer,
        **variant['policy_kwargs']
    )
    es = OUStrategy(
        action_space=env.action_space,
        **variant['es_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_kwargs']
    )
    qf_criterion = variant['qf_criterion_class']()
    ddpg_tdm_kwargs = variant['ddpg_tdm_kwargs']
    ddpg_tdm_kwargs['ddpg_kwargs']['qf_criterion'] = qf_criterion
    ddpg_tdm_kwargs['tdm_kwargs']['tdm_normalizer'] = tdm_normalizer
    algorithm = TdmDdpg(
        env,
        qf=qf,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['ddpg_tdm_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-ddpg-tdm-launch"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "tdm-example-cheetah"

    # noinspection PyTypeChecker
    variant = dict(
        ddpg_tdm_kwargs=dict(
            base_kwargs=dict(
                num_epochs=500,
                num_steps_per_epoch=100,
                num_steps_per_eval=1000,
                max_path_length=99,
                num_updates_per_env_step=25,
                batch_size=128,
                discount=1,
                reward_scale=100,
            ),
            tdm_kwargs=dict(
                max_tau=10,
                num_pretrain_paths=20,
            ),
            ddpg_kwargs=dict(
                tau=0.001,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
            ),
        ),
        her_replay_buffer_kwargs=dict(
            max_size=int(1E6),
        ),
        qf_kwargs=dict(
            hidden_sizes=[300, 300],
            structure='norm_difference',
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        es_kwargs=dict(
            theta=0.1,
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        qf_criterion_class=HuberLoss,
        algorithm="DDPG-TDM",
    )
    for i in range(n_seeds):
        run_experiment(
            experiment,
            mode=mode,
            exp_prefix=exp_prefix,
            variant=variant,
        )
