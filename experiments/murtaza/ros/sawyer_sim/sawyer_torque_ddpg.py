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
from sawyer_control.sawyer_reaching import SawyerJointSpaceReachingEnv


def experiment(variant):
    env_params = variant['env_params']
    env = SawyerJointSpaceReachingEnv(**env_params)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
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
            batch_size=64,
            max_path_length=100,
            num_updates_per_env_step=4,
        ),
        es_kwargs=dict(
            theta=0.1,
            max_sigma=0.25,
            min_sigma=0.25,
        ),
        env_params=dict(
            action_mode='torque',
            reward='norm',
            reward_magnitude=1,
        )
    )
    search_space = {
        'algo_params.reward_scale': [
            1,
            # 10,
            # 100,
            # 1000,
        ],
        'algo_params.num_updates_per_env_step': [
            5,
            # 10,
            15,
            # 20,
            25,
        ],
        'batch_size':[
            64,
            # 128,
            # 256,
            512,
        ],
        'env_params.randomize_goal_on_reset': [
            # True,
            False,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    n_seeds = 3
    for variant in sweeper.iterate_hyperparameters():
        exp_prefix = 'joint_space_reaching_test'
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
            )
