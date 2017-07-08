from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
from os.path import exists
from railrl.envs.mujoco.sawyer_env import SawyerEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
import ipdb

#test if we need to set an action and observation space in sawyer env
def example(variant):
    env = SawyerEnv()
    es = OUStrategy(
        max_sigma=0.05,
        min_sigma=0.05,
        action_space=env.action_space,
    )
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
    use_new_version = variant['use_new_version']
    algorithm = DDPG(
        env,
        es,
        qf=qf,
        policy=policy,
        num_epochs=30,
        batch_size=128,
        use_new_version=use_new_version,
    )
    # ipdb.set_trace()
    algorithm.train()

if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="7-5-ddpg-sawyer-mujoco-fixed-angle-test",
        seed=0,
        mode='here',
        variant={
                'version': 'Original',
                'use_new_version': False,
                },
        use_gpu=False,
    )
