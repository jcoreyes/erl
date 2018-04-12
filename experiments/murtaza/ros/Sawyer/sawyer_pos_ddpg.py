from railrl.envs.wrappers import NormalizedBoxEnv
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
    env = NormalizedBoxEnv(SawyerXYZReachingEnv())
    es = OUStrategy(action_space=env.action_space)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    import ipdb; ipdb.set_trace()
    qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
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
            num_epochs=100,
            num_steps_per_epoch=500,
            num_steps_per_eval=500,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            render=False,
        ),
        env_params=dict(
            desired = [0.79582924, -0.17266372, 0.09573399],
            safety_box=True,
            action_mode='position',
        )
    )
    n_seeds = 1
    exp_prefix = 'sawyer_position_control_ddpg'
    mode = 'here_no_doodad'
    for i in range(n_seeds):
        run_experiment(
            experiment,
            mode=mode,
            exp_prefix=exp_prefix,
            variant=variant,
        )
