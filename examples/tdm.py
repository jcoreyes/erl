import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.point2d import MultitaskPoint2DEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import setup_logger
from railrl.state_distance.tdm_ddpg import TdmDdpg
from railrl.state_distance.tdm_networks import TdmPolicy, \
    TdmQf, TdmNormalizer
from railrl.torch.modules import HuberLoss


def experiment(variant):
    vectorized = variant['ddpg_tdm_kwargs']['tdm_kwargs']['vectorized']
    norm_order = variant['ddpg_tdm_kwargs']['tdm_kwargs']['norm_order']
    max_tau = variant['ddpg_tdm_kwargs']['tdm_kwargs']['max_tau']

    env = NormalizedBoxEnv(MultitaskPoint2DEnv())
    tdm_normalizer = TdmNormalizer(
        env,
        vectorized,
        max_tau=max_tau,
        **variant['tdm_normalizer_kwargs']
    )
    qf = TdmQf(
        env=env,
        vectorized=vectorized,
        norm_order=norm_order,
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
    variant['ddpg_tdm_kwargs']['ddpg_kwargs']['qf_criterion'] = HuberLoss()
    variant['ddpg_tdm_kwargs']['tdm_kwargs']['tdm_normalizer'] = tdm_normalizer
    algorithm = TdmDdpg(
        env,
        qf=qf,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['ddpg_tdm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    num_epochs = 100
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 100

    # noinspection PyTypeChecker
    variant = dict(
        ddpg_tdm_kwargs=dict(
            base_kwargs=dict(
                num_epochs=num_epochs,
                num_steps_per_epoch=num_steps_per_epoch,
                num_steps_per_eval=num_steps_per_eval,
                max_path_length=max_path_length,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
                collection_mode='online',
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=True,
                norm_order=1,
                cycle_taus_for_rollout=True,
                max_tau=10,
                normalize_distance=False,
                terminate_when_goal_reached=False,
                reward_type='distance',
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
            num_goals_to_sample=4,
        ),
        qf_kwargs=dict(
            structure='norm_difference',
            hidden_sizes=[300, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        es_kwargs=dict(
            theta=0.3,
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        version="DDPG-TDM",
        algorithm="DDPG-TDM",
        tdm_normalizer_kwargs=dict(
            normalize_tau=False,
            log_tau=False,
        ),
    )
    setup_logger('tdm-example', variant=variant)
    experiment(variant)
