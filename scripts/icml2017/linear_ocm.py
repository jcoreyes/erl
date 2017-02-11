from railrl.algos.ddpg_ocm import DdpgOcm
from railrl.qfunctions.memory_qfunction import MemoryQFunction
from railrl.exploration_strategies.noop import NoopStrategy
from railrl.envs.memory.continuous_memory_augmented import ContinuousMemoryAugmented
from railrl.envs.memory.one_char_memory import OneCharMemory
from railrl.policies.linear_ocm_policy import LinearOcmPolicy

num_values = 1
H = 1
onehot_dim = num_values + 1

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

memory_dim = env.memory_dim
env_action_dim = env.wrapped_env.action_space.flat_dim
es = NoopStrategy()
qf = MemoryQFunction(
    name_or_scope="critic",
    env_spec=env.spec,
)
algorithm = DdpgOcm(
    env,
    es,
    policy,
    qf,
)
algorithm.train()
