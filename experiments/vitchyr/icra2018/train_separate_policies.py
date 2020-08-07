import random
import numpy as np

from rlkit.envs.mujoco.pusher3dof import PusherEnv3DOF
from rlkit.envs.mujoco.pusher_avoider_3dof import PusherAvoiderEnv3DOF
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from rlkit.torch.ddpg import DDPG
import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.naf import NafPolicy, NAF

from rllab.envs.normalized_env import normalize


def experiment(variant):
    env = variant['env_class'](**variant['env_params'])
    env = normalize(env)
    es = OUStrategy(
        action_space=env.action_space,
        **variant['es_params']
    )
    algo_class = variant['algo_class']
    algo_params = variant['algo_params']
    hidden_size = variant['hidden_size']
    if algo_class == DDPG:
        # algo_params.pop('naf_policy_learning_rate')
        qf = FeedForwardQFunction(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            hidden_size,
            hidden_size,
        )
        policy = FeedForwardPolicy(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            hidden_size,
            hidden_size,
        )
        algorithm = DDPG(
            env,
            exploration_strategy=es,
            qf=qf,
            policy=policy,
            **variant['algo_params']
        )
    elif algo_class == NAF:
        algo_params.pop('qf_learning_rate')
        # algo_params.pop('policy_learning_rate')
        qf = NafPolicy(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            hidden_size,
        )
        algorithm = NAF(
            env,
            policy=qf,
            exploration_strategy=es,
            **variant['algo_params']
        )
    else:
        raise Exception("Invalid algo class: {}".format(algo_class))
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == '__main__':
    num_configurations = 1  # for random mode
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-separate-policies"
    version = "Dev"
    run_mode = "none"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "pusher-avoid-hardcoded-naf"
    # version = "Dev"
    run_mode = 'grid'

    use_gpu = True
    if mode != "here":
        use_gpu = False

    snapshot_mode = "last"
    snapshot_gap = 10
    periodic_sync_interval = 600  # 10 minutes
    variant = dict(
        version=version,
        algo_params=dict(
            num_epochs=501,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1500,
            use_soft_update=True,
            tau=1e-3,
            batch_size=128,
            max_path_length=300,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            # naf_policy_learning_rate=1e-4,
            # render=True,
        ),
        algo_class=NAF,
        # algo_class=DDPG,
        env_class=PusherAvoiderEnv3DOF,
        env_params=dict(
            task='both',
        ),
        hidden_size=400,
        es_params=dict(
            min_sigma=None,
            max_sigma=0.2,
        )
        # env_class=PusherEnv3DOF,
        # env_params=dict(
        #     goal=(1, -1),
        # ),
    )
    if run_mode == 'grid':
        search_space = {
            # 'algo_params.use_soft_update': [True, False],
            'hidden_size': [100, 400],
            # 'es_params.max_sigma': [0.1, 0.3, 0.5, 1],
            # 'env_params.hit_penalty': [0.05, 0.1, 0.5, 1],
            'env_params.task': [
                'push',
                'avoid',
                'both',
            ],
            'env_params.init_config': list(range(5)),
            # 'env_params.goal': [
            #     (-1, -1),
            #     (0, -1),
            #     (1, -1),
            #     (-1, np.nan),
            #     (0, np.nan),
            #     (1, np.nan),
            #     (np.nan, -1),
            # ]
        }
        sweeper = hyp.DeterministicHyperparameterSweeper(
            search_space, default_parameters=variant,
        )
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    use_gpu=use_gpu,
                    sync_s3_log=True,
                    sync_s3_pkl=True,
                    snapshot_mode=snapshot_mode,
                    snapshot_gap=snapshot_gap,
                    periodic_sync_interval=periodic_sync_interval,
                )
    elif run_mode == 'custom_grid':
        for exp_id, (
                goal,
                version,
        ) in enumerate([
            # ((-1, -1), 'bottom-left'),
            # ((0, -1), 'bottom-middle'),
            # ((1, -1), 'bottom-right'),
            # ((-1, np.nan), 'left'),
            # ((0, np.nan), 'middle'),
            # ((1, np.nan), 'right'),
            # ((np.nan, -1), 'bottom'),
        ]):
            variant['version'] = version
            variant['env_params']['goal'] = goal
            new_exp_prefix = "{}_{}".format(exp_prefix, version)
            for _ in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    exp_prefix=new_exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    use_gpu=use_gpu,
                    sync_s3_log=True,
                    sync_s3_pkl=True,
                    snapshot_mode=snapshot_mode,
                    snapshot_gap=snapshot_gap,
                    periodic_sync_interval=periodic_sync_interval,
                )
    else:
        for _ in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=0,
                use_gpu=use_gpu,
                sync_s3_log=True,
                sync_s3_pkl=True,
                snapshot_mode=snapshot_mode,
                snapshot_gap=snapshot_gap,
                periodic_sync_interval=periodic_sync_interval,
            )
