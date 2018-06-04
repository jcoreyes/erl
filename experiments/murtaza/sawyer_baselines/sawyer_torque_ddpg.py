from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.ddpg.ddpg import DDPG
import railrl.torch.pytorch_util as ptu
from sawyer_control.sawyer_reaching import SawyerXYZReachingEnv
def experiment(variant):
    env_params = variant['env_params']
    env = SawyerXYZReachingEnv(**env_params)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = FlattenMlp(
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
            num_steps_per_epoch=500,
            num_steps_per_eval=500,
            use_soft_update=True,
            max_path_length=100,
            render=False,
            normalize_env=False,
            train_on_eval_paths=True,
            num_updates_per_env_step=1,
            collection_mode='online',
            reward_scale=1,
        ),
        es_kwargs=dict(
            theta=0.1,
            max_sigma=.25,
            min_sigma=.25,
        ),
        env_params=dict(
            action_mode='torque',
            reward='norm',
            randomize_goal_on_reset=False,
        )
    )
    n_seeds = 1
    exp_prefix = 'sawyer_torque_ddpg_xyz_reaching_baseline'
    mode = 'here_no_doodad'
    for i in range(n_seeds):
        run_experiment(
            experiment,
            mode=mode,
            exp_prefix=exp_prefix,
            variant=variant,
        )