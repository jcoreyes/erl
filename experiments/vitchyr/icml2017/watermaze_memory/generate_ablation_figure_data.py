"""
Generate data for ablation analysis for ICML 2017 workshop paper.
"""
import random

from torch.nn import functional as F

from rlkit.envs.pygame.water_maze import (
    WaterMazeMemory,
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import (
    run_experiment,
)
from rlkit.launchers.memory_bptt_launchers import bptt_ddpg_launcher
from rlkit.pythonplusplus import identity
from rlkit.memory_states.qfunctions import MemoryQFunction
from rlkit.torch.rnn import GRUCell

if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "7-20-dev-generate-ablation-figure-data"
    run_mode = 'none'

    n_seeds = 5
    mode = "ec2"
    exp_prefix = "7-21-generate-ablation-figure-data-sl5-and-15"

    use_gpu = True
    if mode != "here":
        use_gpu = False

    H = 25
    subtraj_length = None
    num_steps_per_iteration = 1000
    num_steps_per_eval = 1000
    num_iterations = 100
    batch_size = 100
    memory_dim = 100
    version = "Our Method"

    # noinspection PyTypeChecker
    variant = dict(
        memory_dim=memory_dim,
        env_class=WaterMazeMemory,
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
        ),
        qf_class=MemoryQFunction,
        qf_params=dict(
            output_activation=identity,
            fc1_size=400,
            fc2_size=300,
            ignore_memory=False,
        ),
        policy_params=dict(
            fc1_size=400,
            fc2_size=300,
            cell_class=GRUCell,
            output_activation=F.tanh,
            only_one_fc_for_action=False,
        ),
        es_params=dict(
            env_es_class=OUStrategy,
            env_es_params=dict(
                max_sigma=1,
                min_sigma=None,
            ),
            memory_es_class=OUStrategy,
            memory_es_params=dict(
                max_sigma=1,
                min_sigma=None,
            ),
        ),
        version=version,
    )
    # for subtraj_length in [1, 10, 20, 25]:
    for subtraj_length in [5, 15]:
        variant['algo_params']['subtraj_length'] = subtraj_length
        for exp_id, (
                do_not_load_initial_memories,
                qf_ignore_memory,
                version,
        ) in enumerate([
            (True, True, "No Memory States (Truncated BPTT)"),
            (True, False, "No Memory State for Critic"),
            (False, True, "Memory States Not Loaded"),
            (False, False, "Our Method"),
        ]):
            variant['qf_params']['ignore_memory'] = qf_ignore_memory
            variant['algo_params']['do_not_load_initial_memories'] = (
                do_not_load_initial_memories
            )
            variant['version'] = version
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
