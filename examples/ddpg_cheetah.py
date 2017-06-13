"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from railrl.algos.ddpg import DDPG
from os.path import exists
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
import joblib
import tensorflow as tf

def example(variant):
    #cool variant stuff!
    load_policy_file = variant.get('load_policy_file', None)
    if load_policy_file is not None and exists(load_policy_file):
        with tf.Session():
            data = joblib.load(load_policy_file)
            policy = data['policy']
            qf = data['qf']
            replay_buffer=data['pool']
        env = HalfCheetahEnv()
        es = OUStrategy(env_spec=env.spec)
        use_new_version = variant['use_new_version']
        algorithm = DDPG(
            env,
            es,
            policy,
            qf,
            n_epochs=30,
            batch_size=1024,
            replay_pool=replay_buffer,
            use_new_version=use_new_version,
        )
        algorithm.train()
    else:
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
            n_epochs=2,
            batch_size=1024,
            use_new_version=use_new_version,
        )
        algorithm.train()


if __name__ == "__main__":
	# for i in range(10):
	# 	run_experiment(
 #        	example,
 #        	exp_prefix="ddpg-half-cheetah-modified",
 #        	seed=i,
 #        	mode='ec2',
 #        	variant={
 #        	    'version': 'Original',
 #        	    'use_new_version': True,
 #        	}
 #        )
 	run_experiment(
        	example,
        	exp_prefix="ddpg-half-cheetah-6-13",
        	seed=0,
        	mode='here',
        	variant={
        	    'version': 'Original',
        	    'use_new_version': False,
                'load_policy_file': '/home/murtaza/Documents/rllab/data/local/ddpg-half-cheetah-6-13/ddpg-half-cheetah-6-13_2017_06_13_00_43_37_0000--s-0/params.pkl'
    		},
    		snapshot_mode='last',
    	)
