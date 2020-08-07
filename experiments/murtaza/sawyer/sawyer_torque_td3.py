from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3
import rlkit.torch.pytorch_util as ptu
from sawyer_control.sawyer_reaching import SawyerXYZReachingEnv
import rlkit.misc.hyperparameter as hyp
import ray
ray.init()
def experiment(variant):
    env_params = variant['env_params']
    env = SawyerXYZReachingEnv(**env_params)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    hidden_size = variant['hidden_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[hidden_size, hidden_size],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[hidden_size, hidden_size],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[hidden_size, hidden_size],
    )
    es = OUStrategy(
        action_space=env.action_space,
        **variant['es_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TD3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=50,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=64,
            max_path_length=100,
            discount=0.99,
            train_on_eval_paths=True,
            replay_buffer_size=int(1E6),
            normalize_env=False,
        ),
        es_kwargs=dict(
            max_sigma=0.25,
            min_sigma=0.25,
        ),
        env_params=dict(
            action_mode='torque',
            reward='norm',
        ),
        hidden_size=100,
    )
    search_space = {
        'algo_kwargs.num_updates_per_env_step': [
            1,
            5,
        ],
        'env_params.randomize_goal_on_reset': [
            True,
        ],
        'algo_kwargs.batch_size':[
            512,
        ],
        'hidden_size':[
            100,
            150,
        ],
        'algo_kwargs.reward_scale':[
            1,
        ],
        'algo_kwargs.collection_mode':[
            'online-parallel',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    for variant in sweeper.iterate_hyperparameters():
        n_seeds = 1
        exp_prefix = 'sawyer_torque_td3_xyz_reaching_varying_ee_parallel'
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
            )
