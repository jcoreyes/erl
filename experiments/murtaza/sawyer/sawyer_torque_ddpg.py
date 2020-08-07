from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.ddpg.ddpg import DDPG
import rlkit.torch.pytorch_util as ptu
from sawyer_control.sawyer_reaching import SawyerXYZReachingEnv
import rlkit.misc.hyperparameter as hyp
import ray
#ray.init()
def experiment(variant):
    env_params = variant['env_params']
    env = SawyerXYZReachingEnv(**env_params)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[100, 100],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[100, 100],
    )
    es = OUStrategy(
        action_space=env.action_space,
        **variant['es_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
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
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            max_path_length=100,
            render=False,
            normalize_env=False,
            train_on_eval_paths=True,
            #collection_mode='online-parallel',
        ),
        es_kwargs=dict(
            theta=0.1,
            max_sigma=.25,
            min_sigma=.25,
        ),
        env_params=dict(
            action_mode='torque',
            reward='norm'
        )
    )
    search_space = {
        'algo_params.num_updates_per_env_step': [
            1,
            3,
            4,
        ],
        'algo_params.reward_scale': [
            1,
            #100,
        ],
        'env_params.randomize_goal_on_reset': [
            True,
        ],
        'algo_params.batch_size': [
            64,
        ],
        'algo_params.collection_mode':[
            'online'
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    for variant in sweeper.iterate_hyperparameters():
        n_seeds = 1
        exp_prefix = 'sawyer_torque_ddpg_xyz_varying_ee'
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                use_gpu=True,
                exp_prefix=exp_prefix,
                variant=variant,
            )
