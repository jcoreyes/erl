from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.ddpg.ddpg import DDPG
import rlkit.torch.pytorch_util as ptu
from sawyer_control.sawyer_reaching import SawyerXYZReachingEnv
import rlkit.misc.hyperparameter as hyp
import ray
ray.init()

def experiment(variant):
    env_params = variant['env_params']
    es_params = variant['es_params']
    env = SawyerXYZReachingEnv(**env_params)
    es = OUStrategy(action_space=env.action_space, **es_params)
    hidden_sizes = variant['hidden_sizes']

    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[hidden_sizes, hidden_sizes],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[hidden_sizes, hidden_sizes],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    if variant['env_params']['relative_pos_control']:
        variant['algo_params']['max_path_length'] = 3
        variant['algo_params']['num_steps_per_epoch'] = 15
        variant['algo_params']['num_steps_per_eval'] = 15
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )

    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_params=dict(
            num_epochs=50,
            num_steps_per_epoch=50,
            num_steps_per_eval=50,
            use_soft_update=True,
            tau=1e-2,
            batch_size=32,
            max_path_length=10,
            discount=0.9,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            render=False,
            normalize_env=False,
            train_on_eval_paths=True,
        ),
        env_params=dict(
            action_mode='position',
            update_hz=20,
            reward_magnitude=1,
        ),
        es_params=dict(
            theta=.1,
            max_sigma=.25,
            min_sigma=.25
        ),
        hidden_sizes=100,
    )
    search_space = {
        'algo_params.reward_scale': [
            1,
        ],
        'algo_params.num_updates_per_env_step': [
            1,
            5,
        ],
        'algo_params.collection_mode':[
            'online-parallel',
            'online'
        ],
        'env_params.relative_pos_control':[
            False,
            True,
        ],
        'hidden_sizes':[
            100,
            200,
        ],
        'env_params.randomize_goal_on_reset': [
            True,
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    for variant in sweeper.iterate_hyperparameters():
        n_seeds = 1
        exp_prefix = 'sawyer_pos_ddpg'
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
            )
