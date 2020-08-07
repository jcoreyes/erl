"""
Exampling of running DDPG on Double Reacher.
"""
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from rlkit.memory_states.qfunctions import FeedForwardDuelingQFunction
from rlkit.envs.env_utils import gym_env
from rllab.envs.normalized_env import normalize
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.ddpg import DDPG
from os.path import exists
import joblib

def example(variant):
    load_policy_file = variant.get('load_policy_file', None)
    if not load_policy_file == None and exists(load_policy_file):
        data = joblib.load(load_policy_file)
        algorithm = data['algorithm']
        epochs = algorithm.num_epochs - data['epoch']
        algorithm.num_epochs = epochs
        use_gpu = variant['use_gpu']
        if use_gpu and ptu.gpu_enabled():
            algorithm.cuda()
        algorithm.train()
    else:
        es_min_sigma = variant['es_min_sigma']
        es_max_sigma = variant['es_max_sigma']
        num_epochs = variant['num_epochs']
        batch_size = variant['batch_size']
        use_gpu = variant['use_gpu']
        dueling = variant['dueling']

        env = normalize(gym_env('Reacher-v1'))
        es = OUStrategy(
            max_sigma=es_max_sigma,
            min_sigma=es_min_sigma,
            action_space=env.action_space,
        )
        if dueling:
            qf = FeedForwardDuelingQFunction(
                int(env.observation_space.flat_dim),
                int(env.action_space.flat_dim),
                100,
                100,
            )
        else:
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
        algorithm = DDPG(
            env,
            qf,
            policy,
            es,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        if use_gpu:
            algorithm.cuda()
        algorithm.train()

if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="ddpg-reacher-dueling-qf",
        seed=0,
        mode='local',
        variant={
            'version': 'Original',
            'es_min_sigma': .05,
            'es_max_sigma': .05,
            'num_epochs': 50,
            'batch_size': 1024,
            'use_gpu': True,
            'dueling':True,
            'load_policy_file':'/home/murtaza/Documents/rllab/data/local/7-24-reacher-algorithm-restart-test/7-24-ddpg-reacher-algorithm-restart-test_2017_07_24_12_35_59_0000--s-0/params.pkl'
        },
        use_gpu=True,
    )
