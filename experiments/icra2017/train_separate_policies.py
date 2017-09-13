import random
import numpy as np

from railrl.envs.mujoco.pusher3dof import PusherEnv3DOF
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu

from rllab.envs.normalized_env import normalize


def experiment(variant):
    env = PusherEnv3DOF(**variant['env_params'])
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
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
    mode = "ec2"
    exp_prefix = "pusher-3dof-horizontal-l2-top"
    # version = "Dev"
    # run_mode = 'grid'

    use_gpu = True
    if mode != "here":
        use_gpu = False

    snapshot_mode = "last"
    snapshot_gap = 10
    periodic_sync_interval = 600  # 10 minutes
    variant = dict(
        version=version,
        algo_params=dict(
            num_epochs=101,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1500,
            use_soft_update=True,
            tau=1e-3,
            batch_size=128,
            max_path_length=300,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        env_params=dict(
            # goal=(np.nan, -1),
            goal=(1, np.nan),
        ),
    )
    if run_mode == 'grid':
        search_space = {
            # 'algo_params.use_soft_update': [True, False],
            # 'algo_params.tau': [1e-2, 1e-3],
            # 'algo_params.batch_size': [128, 512],
            # 'algo_params.scale_reward': [0.1, 1, 10, 100, 1000],
            # 'env_params.goal': [
            #     (0, np.nan),
            #     (np.nan, -1)
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
