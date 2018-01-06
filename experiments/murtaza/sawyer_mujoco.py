from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from railrl.torch.ddpg import DDPG
from os.path import exists
from railrl.envs.mujoco.sawyer_env import SawyerEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
import ipdb

#test if we need to set an action and observation space in sawyer env
def example(variant):
    loss = variant['loss']
    experiment=variant['experiment']
    env = SawyerEnv(loss=loss)
    es = OUStrategy(
        max_sigma=1,
        min_sigma=1,
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
    use_target_policy = variant['use_target_policy']
    algorithm = DDPG(
        env,
        es,
        qf=qf,
        policy=policy,
        num_epochs=30,
        batch_size=1024,
        use_target_policy=use_target_policy,
        render=True,
    )
    # ipdb.set_trace()
    algorithm.train()

if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="7-12-ddpg-sawyer-mujoco-fixed-angle-updated-log-diagnostics",
        seed=0,
        mode='here',
        variant={
                'version': 'Original',
                'use_target_policy': True,
                'loss':'huber',
                'experiment':'joint_angle_fixed',
                },
        use_gpu=True,
    )
