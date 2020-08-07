"""
Exampling of running DDPG on HalfCheetah.
"""
from os.path import exists

import joblib
import tensorflow as tf

from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.qfunctions.nn_qfunction import FeedForwardCritic
from rlkit.tf.ddpg import DDPG
from rlkit.tf.policies.nn_policy import FeedForwardPolicy
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv


def example(variant):
    load_policy_file = variant.get('load_policy_file', None)
    if load_policy_file is not None and exists(load_policy_file):
        with tf.Session():
            data = joblib.load(load_policy_file)
            print(data)
            policy = data['policy']
            qf = data['qf']
            replay_buffer=data['pool']
        env = HalfCheetahEnv()
        es = OUStrategy(action_space=env.action_space)
        use_new_version = variant['use_new_version']
        algorithm = DDPG(
            env,
            es,
            policy,
            qf,
            n_epochs=2,
            batch_size=1024,
            replay_pool=replay_buffer,
            use_new_version=use_new_version,
        )
        algorithm.train()
    else:
        env = HalfCheetahEnv()
        es = OUStrategy(action_space=env.action_space)
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
            n_epochs=2,
            batch_size=1024,
            use_new_version=use_new_version,
        )
        algorithm.train()


if __name__ == "__main__":
    using_ec2 = False
    seeds = 1
    if(using_ec2):
        for i in range(seeds):
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
    else:
        run_experiment(
        	example,
        	exp_prefix="ddpg-half-cheetah-6-14-TEST-DELETE",
        	seed=0,
        	mode='here',
        	variant={
        	    'version': 'Original',
        	    'use_new_version': False,
                # 'load_policy_file': '/home/murtaza/Documents/rllab/data/local/ddpg-half-cheetah-6-13/ddpg-half-cheetah-6-13_2017_06_13_09_05_31_0000--s-0/params.pkl'
    		},
    		snapshot_mode='last',
    	)