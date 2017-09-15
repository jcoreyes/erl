import random
import numpy as np

from railrl.envs.mujoco.pusher3dof import PusherEnv3DOF
from railrl.envs.mujoco.pusher_avoider_3dof import PusherAvoiderEnv3DOF
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.torch.naf import NafPolicy, NAF

from rllab.envs.normalized_env import normalize


def experiment(variant):
    env = variant['env_class'](**variant['env_params'])
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    algo_class = variant['algo_class']
    algo_params = variant['algo_params']
    if algo_class == DDPG:
        algo_params.pop('naf_policy_learning_rate')
        qf = FeedForwardQFunction(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            100,
            100,
        )
        policy = FeedForwardPolicy(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            100,
            100,
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
        algo_params.pop('policy_learning_rate')
        qf = NafPolicy(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            100,
        )
        algorithm = NAF(
            env,
            naf_policy=qf,
            exploration_strategy=es,
            **variant['algo_params']
        )
    else:
        raise Exception("Invalid algo class: {}".format(algo_class))
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == '__main__':
    num_configurations = 1  # for random mode
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-separate-policies"
    version = "Dev"
    run_mode = "none"

    n_seeds = 3
    # mode = "ec2"
    # exp_prefix = "pusher-3dof-reacher-bottom-right"
    # version = "Dev"
    run_mode = 'custom_grid'

    use_gpu = True
    if mode != "here":
        use_gpu = False

    snapshot_mode = "last"
    snapshot_gap = 20
    periodic_sync_interval = 600  # 10 minutes
    variant = dict(
        version=version,
        algo_params=dict(
            num_epochs=201,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1500,
            use_soft_update=True,
            tau=1e-3,
            batch_size=128,
            max_path_length=300,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            naf_policy_learning_rate=1e-4,
        ),
        algo_class=NAF,
        # env_class=PusherAvoiderEnv3DOF,
        # env_params=dict(
        #     task='push',
        # ),
        env_class=PusherEnv3DOF,
        env_params=dict(
            goal=(1, -1),
        ),
    )
    if run_mode == 'grid':
        search_space = {
            # 'algo_params.use_soft_update': [True, False],
            # 'algo_params.tau': [1e-2, 1e-3],
            # 'algo_params.batch_size': [128, 512],
            # 'env_params.hit_penalty': [0.05, 0.1, 1, 5],
            # 'env_params.task': [
            #     'push',
            #     'avoid',
            #     'both',
            # ]
            'env_params.goal': [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, np.nan),
                (0, np.nan),
                (1, np.nan),
                (np.nan, -1),
            ]
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
            ((-1, -1), 'bottom-left'),
            ((0, -1), 'bottom-middle'),
            ((1, -1), 'bottom-right'),
            ((-1, np.nan), 'left'),
            ((0, np.nan), 'middle'),
            ((1, np.nan), 'right'),
            ((np.nan, -1), 'bottom'),
        ]):
            variant['version'] = version
            variant['env_params']['goal'] = goal
            new_exp_prefix = "{}_{}".format(exp_prefix, version)
            for _ in range(n_seeds):
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
