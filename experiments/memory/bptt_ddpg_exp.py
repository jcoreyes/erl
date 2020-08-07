"""
Try the PyTorch version of BPTT DDPG on HighLow env.
"""
import random

from rlkit.envs.pygame.water_maze import (
    WaterMazeMemory,
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import (
    run_experiment,
)
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.memory_bptt_launchers import bptt_ddpg_launcher
from rlkit.pythonplusplus import identity
from rlkit.memory_states.qfunctions import MemoryQFunction
from rlkit.torch.rnn import LSTMCell, BNLSTMCell, GRUCell

from torch.nn import functional as F
import rlkit.torch.pytorch_util as ptu


if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "7-19-dev-bptt-ddpg-check"
    run_mode = 'none'
    version = "Our Method"

    n_seeds = 1
    mode = "ec2"
    exp_prefix = "7-20-timeit-c4xlarge-correct-price"
    version = "Our Method - c4.xlarge-correct-price"

    # run_mode = 'grid'
    num_configurations = 25
    use_gpu = True
    if mode != "here":
        use_gpu = False

    H = 25
    subtraj_length = 5
    num_steps_per_iteration = 100
    num_steps_per_eval = 1000
    num_iterations = 100
    batch_size = 100
    memory_dim = 100

    # noinspection PyTypeChecker
    variant = dict(
        memory_dim=memory_dim,
        # env_class=WaterMaze,
        # env_class=WaterMazeEasy,
        # env_class=WaterMazeMemory1D,
        env_class=WaterMazeMemory,
        # env_class=WaterMazeHard,
        # env_class=HighLow,
        env_params=dict(
            horizon=H,
            give_time=True,
            # action_l2norm_penalty=0,
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
            write_policy_optimizes='both',
            action_policy_learning_rate=0.001,
            write_policy_learning_rate=0.0005,
            qf_learning_rate=0.002,
            max_path_length=H,
            refresh_entire_buffer_period=None,
            save_new_memories_back_to_replay_buffer=True,
            write_policy_weight_decay=0,
            action_policy_weight_decay=0,
            do_not_load_initial_memories=False,
            save_memory_gradients=False,
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
            ignore_memory=False,
        ),
        policy_params=dict(
            fc1_size=400,
            fc2_size=300,
            cell_class=GRUCell,
            # cell_class=RWACell,
            # cell_class=BNLSTMCell,
            # cell_class=LSTMCell,
            output_activation=F.tanh,
            # output_activation=ptu.clip1,
            only_one_fc_for_action=True,
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
        for fc1, fc2 in [
            (32, 32),
            (400, 300),
        ]:
            search_space = {
                # 'algo_params.qf_learning_rate': [1e-3, 1e-5],
                # 'algo_params.action_policy_learning_rate': [1e-3, 1e-5],
                # 'algo_params.write_policy_learning_rate': [1e-5, 1e-7],
                # 'algo_params.do_not_load_initial_memories': [True, False],
                'algo_params.write_policy_optimizes': ['bellman', 'both'],
                # 'algo_params.refresh_entire_buffer_period': [None, 1],
                # 'es_params.memory_es_params.max_sigma': [0, 1],
                # 'qf_params.ignore_memory': [True, False],
                # 'policy_params.hidden_init': [init.kaiming_normal, ptu.fanin_init],
                'policy_params.output_activation': [F.tanh, ptu.clip1],
                'policy_params.cell_class': [LSTMCell, BNLSTMCell, GRUCell],
                'policy_params.only_one_fc_for_action': [True, False],
                # 'algo_params.subtraj_length': [1, 5, 10, 15, 20, 25],
                # 'algo_params.bellman_error_loss_weight': [0.1, 1, 10, 100, 1000],
                # 'algo_params.tau': [1, 0.1, 0.01, 0.001],
                # 'env_params.give_time': [True, False],
                # 'algo_params.discount': [1, .9, .5, 0],
                # 'env_params.action_l2norm_penalty': [0, 1e-3, 1e-2, 1e-1, 1, 10],
            }
            variant['policy_params']['fc1_size'] = fc1
            variant['policy_params']['fc2_size'] = fc2
            sweeper = hyp.DeterministicHyperparameterSweeper(
                search_space, default_parameters=variant,
            )
            for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
                for i in range(n_seeds):
                    run_experiment(
                        bptt_ddpg_launcher,
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
                    bptt_ddpg_launcher,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    if run_mode == 'random':
        for (
            rnn_cell,
            output_activation,
        ) in [
            (LSTMCell, F.tanh),
            (LSTMCell, ptu.clip1),
            (GRUCell, F.tanh),
            (GRUCell, ptu.clip1),
        ]:
            variant['policy_params']['cell_class'] = rnn_cell
            variant['policy_params']['output_activation'] = output_activation
            hyperparameters = [
                hyp.LogIntParam('memory_dim', 4, 400),
                hyp.LogFloatParam('algo_params.qf_learning_rate', 1e-5, 1e-2),
                hyp.LogFloatParam(
                    'algo_params.write_policy_learning_rate', 1e-5, 1e-3
                ),
                hyp.LogFloatParam(
                    'algo_params.action_policy_learning_rate', 1e-5, 1e-3
                ),
                # hyp.EnumParam(
                #     'algo_params.action_policy_optimize_bellman', [True, False],
                # ),
                # hyp.EnumParam(
                #     'algo_params.use_action_policy_params_for_entire_policy',
                #     [True, False],
                # ),
                # hyp.EnumParam(
                #     'algo_params.write_policy_optimizes', ['both', 'qf', 'bellman']
                # ),
                # hyp.EnumParam(
                #     'policy_params.cell_class', [GRUCell, LSTMCell],
                # ),
                # hyp.EnumParam(
                #     'es_params.memory_es_params.max_sigma', [0, 0.1, 1],
                # ),
                # hyp.LogFloatParam(
                #     'algo_params.write_policy_weight_decay', 1e-5, 1e2,
                # ),
                # hyp.LogFloatParam(
                #     'algo_params.action_policy_weight_decay', 1e-5, 1e2,
                # ),
                # hyp.EnumParam(
                #     'policy_params.output_activation', [F.tanh, ptu.clip1],
                # ),
                # hyp.EnumParam(
                #     'es_params.memory_es_class', [OUStrategy, NoopStrategy],
                # ),
                # hyp.LogFloatParam(
                #     'env_params.action_l2norm_penalty', 1e-2, 10,
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
                        bptt_ddpg_launcher,
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
                bptt_ddpg_launcher,
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
