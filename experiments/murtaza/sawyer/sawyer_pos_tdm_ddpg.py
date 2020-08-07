from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.envs.multitask.sawyer_env_v2 import MultiTaskSawyerXYZReachingEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.tdm_ddpg import TdmDdpg
from rlkit.state_distance.tdm_networks import TdmNormalizer, TdmQf, TdmPolicy
from rlkit.torch.networks.experimental import HuberLoss
import rlkit.torch.pytorch_util as ptu
import rlkit.misc.hyperparameter as hyp

def experiment(variant):
    env_params = variant['env_params']
    env = MultiTaskSawyerXYZReachingEnv(**env_params)
    max_tau = variant['ddpg_tdm_kwargs']['tdm_kwargs']['max_tau']
    tdm_normalizer = TdmNormalizer(
        env,
        vectorized=True,
        max_tau=max_tau,
    )
    qf = TdmQf(
        env=env,
        vectorized=True,
        norm_order=2,
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
    variant = dict(
        ddpg_tdm_kwargs=dict(
            base_kwargs=dict(
                num_epochs=50,
                num_steps_per_epoch=50,
                num_steps_per_eval=50,
                max_path_length=10,
                num_updates_per_env_step=4,
                batch_size=64,
                discount=0.9
            ),
            tdm_kwargs=dict(
                max_tau=10,
                num_pretrain_paths=0,
                vectorized=False,
            ),
            ddpg_kwargs=dict(
                tau=0.001,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
            ),
        ),
        her_replay_buffer_kwargs=dict(
            max_size=int(2E4),
        ),
        qf_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        es_kwargs=dict(
            theta=0.1,
            max_sigma=0.25,
            min_sigma=0.25,
        ),
        qf_criterion_class=HuberLoss,
        env_params=dict(
			desired=[0.59, 0.1, 0.4],
            action_mode='pos',
            reward_magnitude=10,
        )
    )
    search_space = {

        'algo_params.max_path_length': [
			5,
			10,
			15,
		],
		'algo_params.num_updates_per_env_step': [
			5,
			10,
			15,
		],
		'env_params.randomize_goal_on_reset': [
			True,
		],
	}
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for variant in sweeper.iterate_hyperparameters():
        n_seeds = 1
        exp_prefix = 'sawyer_tdm_ddpg_pos'
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
            )