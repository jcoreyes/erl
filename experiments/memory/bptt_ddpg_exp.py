"""
Try the PyTorch version of BPTT DDPG on HighLow env.
"""
import random

from railrl.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from railrl.envs.memory.high_low import HighLow
from railrl.envs.pygame.water_maze import (
    WaterMaze,
    WaterMazeEasy,
    WaterMazeMemory,
    WaterMaze1D,
    WaterMazeEasy1D,
    WaterMazeMemory1D,
)
from railrl.exploration_strategies.noop import NoopStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import (
    run_experiment,
)
import railrl.misc.hyperparameter as hyp
from railrl.policies.torch import MemoryPolicy, RWACell
from railrl.pythonplusplus import identity
from railrl.qfunctions.torch import MemoryQFunction, RecurrentMemoryQFunction
from railrl.torch.rnn import LSTMCell, BNLSTMCell, GRUCell
from railrl.torch.bptt_ddpg_rq import BpttDdpgRecurrentQ

from torch.nn import init
import railrl.torch.pytorch_util as ptu


def experiment(variant):
    from railrl.torch.bptt_ddpg import BpttDdpg
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.exploration_strategies.product_strategy import ProductStrategy
    seed = variant['seed']
    algo_params = variant['algo_params']
    memory_dim = variant['memory_dim']
    rnn_cell = variant['policy_params']['cell_class']
    memory_dim -= memory_dim % rnn_cell.state_num_split()
    env_class = variant['env_class']
    env_params = variant['env_params']
    memory_aug_params = variant['memory_aug_params']

    qf_class = variant['qf_class']
    qf_params = variant['qf_params']
    policy_params = variant['policy_params']

    es_params = variant['es_params']
    env_es_class = es_params['env_es_class']
    env_es_params = es_params['env_es_params']
    memory_es_class = es_params['memory_es_class']
    memory_es_params = es_params['memory_es_params']

    set_seed(seed)
    raw_env = env_class(**env_params)
    env = ContinuousMemoryAugmented(
        raw_env,
        num_memory_states=memory_dim,
        **memory_aug_params
    )
    env_strategy = env_es_class(
        action_space=raw_env.action_space,
        **env_es_params
    )
    write_strategy = memory_es_class(
        action_space=env.memory_state_space,
        **memory_es_params
    )
    es = ProductStrategy([env_strategy, write_strategy])
    qf = qf_class(
        int(raw_env.observation_space.flat_dim),
        int(raw_env.action_space.flat_dim),
        memory_dim,
        **qf_params,
    )
    policy = MemoryPolicy(
        int(raw_env.observation_space.flat_dim),
        int(raw_env.action_space.flat_dim),
        memory_dim=memory_dim,
        **policy_params
    )
    algorithm = BpttDdpg(
        env,
        es,
        qf=qf,
        policy=policy,
        **algo_params
    )
    algorithm.train()


if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "7-8-dev-bptt-ddpg-exp"
    run_mode = 'none'

    n_seeds = 10
    mode = "ec2"
    exp_prefix = "7-8-bptt-ddpg-water-maze-memory-sweep-subtraj-length"

    run_mode = 'grid'
    num_configurations = 500
    use_gpu = True
    if mode != "here":
        use_gpu = False

    H = 25
    subtraj_length = 25
    num_steps_per_iteration = 1000
    num_steps_per_eval = 1000
    num_iterations = 50
    batch_size = 200
    memory_dim = 100
    version = exp_prefix
    version = "Our Method"
    # version = "Our Method - loading but Q does not read mem state"
    # noinspection PyTypeChecker
    variant = dict(
        memory_dim=memory_dim,
        # env_class=WaterMaze,
        # env_class=WaterMazeEasy,
        # env_class=WaterMazeMemory1D,
        env_class=WaterMazeMemory,
        # env_class=HighLow,
        env_params=dict(
            horizon=H,
            give_time=True,
        ),
        memory_aug_params=dict(
            max_magnitude=1,
        ),
        algo_params=dict(
            subtraj_length=subtraj_length,
            batch_size=batch_size,
            num_epochs=num_iterations,
            num_steps_per_epoch=num_steps_per_iteration,
            num_steps_per_eval=num_steps_per_eval,
            discount=0.9,
            use_action_policy_params_for_entire_policy=False,
            action_policy_optimize_bellman=False,
            write_policy_optimizes='bellman',
            action_policy_learning_rate=0.000980014225523977,
            write_policy_learning_rate=0.0005,
            qf_learning_rate=0.002021863834563243,
            max_path_length=H,
            refresh_entire_buffer_period=None,
            save_new_memories_back_to_replay_buffer=True,
            write_policy_weight_decay=0,
            action_policy_weight_decay=0,
            # tau=0.001,
            # use_soft_update=False,
            # target_hard_update_period=300,
        ),
        # qf_class=RecurrentMemoryQFunction,
        qf_class=MemoryQFunction,
        qf_params=dict(
            output_activation=identity,
            # hidden_size=10,
            fc1_size=400,
            fc2_size=300,
        ),
        policy_params=dict(
            fc1_size=400,
            fc2_size=300,
            cell_class=GRUCell,
        ),
        es_params=dict(
            env_es_class=OUStrategy,
            env_es_params=dict(
                max_sigma=1,
                min_sigma=None,
            ),
            # memory_es_class=NoopStrategy,
            memory_es_class=OUStrategy,
            memory_es_params=dict(
                max_sigma=1,
                min_sigma=None,
            ),
        ),
        version=version,
    )
    if run_mode == 'grid':
        search_space = {
            # 'algo_params.qf_learning_rate': [1e-3, 1e-5],
            # 'algo_params.action_policy_learning_rate': [1e-3, 1e-5],
            # 'algo_params.write_policy_learning_rate': [1e-5, 1e-7],
            # 'algo_params.action_policy_optimize_bellman': [True, False],
            # 'algo_params.write_policy_optimizes': ['qf', 'bellman', 'both'],
            # 'algo_params.refresh_entire_buffer_period': [None, 1],
            # 'es_params.memory_es_params.max_sigma': [0, 1],
            # 'qf_params.hidden_init': [init.kaiming_normal, ptu.fanin_init],
            # 'policy_params.hidden_init': [init.kaiming_normal, ptu.fanin_init],
            # 'policy_params.feed_action_to_memory': [False, True],
            # 'policy_params.cell_class': [LSTMCell, BNLSTMCell, RWACell],
            'algo_params.subtraj_length': [1, 15, 25],
            # 'algo_params.bellman_error_loss_weight': [0.1, 1, 10, 100, 1000],
            # 'algo_params.tau': [1, 0.1, 0.01, 0.001],
            # 'env_params.give_time': [True, False],
            # 'algo_params.discount': [1, .9, .5, 0],
        }
        sweeper = hyp.DeterministicHyperparameterSweeper(
            search_space, default_parameters=variant,
        )
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=i,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    if run_mode == 'custom_grid':
        for exp_id, (
            action_policy_optimize_bellman,
            write_policy_optimizes,
            refresh_entire_buffer_period,
        ) in enumerate([
            (True, 'both', 1),
            (False, 'qf', 1),
            (True, 'both', None),
            (False, 'qf', None),
        ]):
            variant['algo_params']['action_policy_optimize_bellman'] = (
                action_policy_optimize_bellman
            )
            variant['algo_params']['write_policy_optimizes'] = (
                write_policy_optimizes
            )
            variant['algo_params']['refresh_entire_buffer_period'] = (
                refresh_entire_buffer_period
            )
            for _ in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    if run_mode == 'random':
        hyperparameters = [
            hyp.LogIntParam('memory_dim', 4, 400),
            hyp.LogFloatParam('algo_params.qf_learning_rate', 1e-5, 1e-2),
            # hyp.LogFloatParam(
            #     'algo_params.write_policy_learning_rate', 1e-6, 1e-3
            # ),
            hyp.LogFloatParam(
                'algo_params.action_policy_learning_rate', 1e-6, 1e-3
            ),
            hyp.EnumParam(
                'algo_params.action_policy_optimize_bellman', [True, False],
            ),
            hyp.EnumParam(
                'algo_params.use_action_policy_params_for_entire_policy',
                [True, False],
            ),
            # hyp.EnumParam(
            #     'algo_params.write_policy_optimizes', ['both', 'qf', 'bellman']
            # ),
            # hyp.EnumParam(
            #     'policy_params.cell_class', [GRUCell, BNLSTMCell, LSTMCell,
            #                                  RWACell],
            # ),
            hyp.EnumParam(
                'es_params.memory_es_params.max_sigma', [0, 0.1, 1],
            ),
            # hyp.LogFloatParam(
            #     'algo_params.write_policy_weight_decay', 1e-5, 1e2,
            # ),
            hyp.LogFloatParam(
                'algo_params.action_policy_weight_decay', 1e-5, 1e2,
            ),
            # hyp.LinearFloatParam(
            #     'algo_params.discount', 0.8, 0.99,
            # ),
        ]
        sweeper = hyp.RandomHyperparameterSweeper(
            hyperparameters,
            default_kwargs=variant,
        )
        for exp_id in range(num_configurations):
            variant = sweeper.generate_random_hyperparameters()
            for _ in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    sync_s3_log=True,
                    sync_s3_pkl=True,
                    periodic_sync_interval=600,
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
                periodic_sync_interval=120,
            )
