"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from railrl.algos.ddpg import DDPG

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv


def example(variant):
    env = HalfCheetahEnv()
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )
    use_new_version = variant['use_new_version']
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        n_epochs=30,
        batch_size=1024,
        use_new_version=use_new_version,
    )
    algorithm.train()


if __name__ == "__main__":
	for i in range(10):
		run_experiment(
        	example,
        	exp_prefix="ddpg-half-cheetah-modified",
        	seed=i,
        	mode='ec2',
        	variant={
        	    'version': 'Original',
        	    'use_new_version': True,
        	}
        )
 	# run_experiment(
  #       	example,
  #       	exp_prefix="ddpg-half-cheetah-normal",
  #       	seed=0,
  #       	mode='here',
  #       	variant={
  #       	    'version': 'Original',
  #       	    'use_new_version': False,
  #   		}
  #   	)
