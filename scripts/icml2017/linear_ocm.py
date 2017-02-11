import tensorflow as tf

from railrl.algos.ddpg_ocm import DdpgOcm
from railrl.qfunctions.memory_qfunction import MemoryQFunction
from railrl.exploration_strategies.noop import NoopStrategy
from railrl.envs.memory.continuous_memory_augmented import ContinuousMemoryAugmented
from railrl.envs.memory.one_char_memory import OneCharMemory
from railrl.policies.linear_ocm_policy import LinearOcmPolicy
from rllab.sampler.utils import rollout

num_values = 1
H = 1
onehot_dim = num_values + 1

sess = tf.InteractiveSession()
with sess.as_default():

    env = OneCharMemory(n=num_values, num_steps=H)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=onehot_dim,
    )

    policy = LinearOcmPolicy(
        name_or_scope="policy",
        memory_and_action_dim=onehot_dim,
        horizon=H,
        env_spec=env.spec,
    )

    sess.run(tf.global_variables_initializer())

    path = rollout(env, policy)
    import ipdb
    ipdb.set_trace()
