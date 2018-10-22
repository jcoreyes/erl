from railrl.envs.multitask.sawyer_env_v2 import MultiTaskSawyerXYZReachingEnv
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.state_distance.tdm_ddpg import TdmDdpg
from railrl.state_distance.tdm_networks import TdmQf, TdmPolicy, TdmNormalizer
from railrl.torch.modules import HuberLoss
import railrl.torch.pytorch_util as ptu
import railrl.misc.hyperparameter as hyp
import copy
def experiment(variant):
    env_params = variant['env_params']
    env = MultiTaskSawyerXYZReachingEnv(env_params)
    tdm_normalizer = TdmNormalizer(
        env,
        vectorized=True,
        max_tau=variant['ddpg_tdm_kwargs']['tdm_kwargs']['max_tau'],
    )
    qf = TdmQf(
        env=env,
        vectorized=True,
        hidden_sizes=[variant['hidden_sizes'], variant['hidden_sizes']],
        structure='norm_difference',
        tdm_normalizer=tdm_normalizer,
    )
    policy = TdmPolicy(
        env=env,
        hidden_sizes=[variant['hidden_sizes'], variant['hidden_sizes']],
        tdm_normalizer=tdm_normalizer,
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
    ddpg_tdm_kwargs = copy.deepcopy(variant['ddpg_tdm_kwargs'])
    ddpg_tdm_kwargs['ddpg_kwargs']['qf_criterion'] = qf_criterion
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
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                batch_size=64,
                discount=1,
                normalize_env=False,
            ),
            tdm_kwargs=dict(
                num_pretrain_paths=0,
                vectorized=True,
                max_tau=15,
            ),
            ddpg_kwargs=dict(
                tau=0.001,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
            ),
        ),
        her_replay_buffer_kwargs=dict(
            max_size=int(2E5),
        ),
        es_kwargs=dict(
            theta=0.1,
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        qf_criterion_class=HuberLoss,
        env_params=dict(
            action_mode='torque',
        ),
        hidden_sizes=100,
    )
    search_space = {
        'ddpg_tdm_kwargs.base_kwargs.num_updates_per_env_step': [
            5,
<<<<<<< HEAD
            4,
        ],
        'ddpg_tdm_kwargs.base_kwargs.reward_scale': [
            1,
=======
        ],
        'ddpg_tdm_kwargs.base_kwargs.reward_scale': [
>>>>>>> shikhar_vae
            10,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.sample_rollout_goals_from': [
            'replay_buffer',
        ],
        'hidden_sizes': [
<<<<<<< HEAD
            50,
            100,
            200,
=======
            100,
>>>>>>> shikhar_vae
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    for variant in sweeper.iterate_hyperparameters():
<<<<<<< HEAD
        n_seeds = 3
        exp_prefix = 'sawyer_torque_tdm_ddpg_xyz_reaching_sweep2'
=======
        n_seeds = 1
        exp_prefix = 'sawyer_torque_tdm_ddpg_xyz_reaching_nupo_5'
>>>>>>> shikhar_vae
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
<<<<<<< HEAD
            )
=======
            )
>>>>>>> shikhar_vae
