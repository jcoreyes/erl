import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.envs.mujoco.sawyer_gripper_env import SawyerXYEnv
from rlkit.envs.mujoco.sawyer_reach_env import SawyerReachXYEnv
from rlkit.state_distance.tdm_networks import TdmPolicy, \
    TdmQf
from rlkit.state_distance.tdm_td3 import TdmTd3
from rlkit.torch.networks.experimental import HuberLoss


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    tdm_normalizer = None
    qf1 = TdmQf(
        env=env,
        vectorized=True,
        tdm_normalizer=tdm_normalizer,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
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
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['td3_kwargs']['qf_criterion'] = qf_criterion
    algo_kwargs['tdm_kwargs']['tdm_normalizer'] = tdm_normalizer
    algorithm = TdmTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-tdm-td3-full-state-sawyer"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "tdm-td3-reach-sweep"

    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=200,
                num_steps_per_epoch=50,
                num_steps_per_eval=1000,
                max_path_length=50,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
                reward_scale=1,
                min_num_steps_before_training=128,
            ),
            tdm_kwargs=dict(
                max_tau=15,
                num_pretrain_paths=0,
            ),
            td3_kwargs=dict(
            ),
        ),
        env_class=SawyerXYEnv,
        env_kwargs=dict(
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
        algorithm="TDM-TD3",
    )

    search_space = {
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [1, 5, 10],
        'algo_kwargs.tdm_kwargs.max_tau': [0, 5, 10],
        'env_class': [SawyerXYEnv, SawyerReachXYEnv],
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
            )
