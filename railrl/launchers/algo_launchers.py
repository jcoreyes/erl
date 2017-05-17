"""
This file contains classic RL launchers. See module docstring for more detail.
"""


#################
# my algorithms #
#################
import joblib
import tensorflow as tf

from railrl.exploration_strategies.action_aware_memory_strategy import \
    ActionAwareMemoryStrategy
from railrl.policies.memory.action_aware_memory_policy import \
    ActionAwareMemoryPolicy


def mem_ddpg_launcher(variant):
    """
    Run DDPG with a memory state policy
    :param variant: Dictionary of dictionary with the following keys:
        - algo_params
        - env_params
        - qf_params
        - policy_params
    :return:
    """
    from railrl.algos.ddpg import DDPG
    from railrl.algos.ddpg_ocm import DdpgOcm
    from railrl.policies.memory.softmax_memory_policy import SoftmaxMemoryPolicy
    from railrl.qfunctions.memory.memory_qfunction import MlpMemoryQFunction
    from railrl.launchers.launcher_util import get_env_settings
    from railrl.core.tf_util import BatchNormConfig
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented,
    )
    from railrl.envs.memory.one_char_memory import OneCharMemory
    from railrl.exploration_strategies.noop import NoopStrategy
    if ('batch_norm_params' in variant
        and variant['batch_norm_params'] is not None):
        bn_config = BatchNormConfig(**variant['batch_norm_params'])
    else:
        bn_config = None
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']

    assert isinstance(env, ContinuousMemoryAugmented)
    memory_dim = env.memory_dim
    env_action_dim = env.wrapped_env.action_space.flat_dim
    es = NoopStrategy()
    qf = MlpMemoryQFunction(
        name_or_scope="critic",
        env_spec=env.spec,
        batch_norm_config=bn_config,
        **variant.get('qf_params', {})
    )
    policy = SoftmaxMemoryPolicy(
        name_or_scope="actor",
        memory_dim=memory_dim,
        env_action_dim=env_action_dim,
        env_spec=env.spec,
        batch_norm_config=bn_config,
        **variant.get('policy_params', {})
    )
    if isinstance(env.wrapped_env, OneCharMemory):
        ddpg_class = DdpgOcm
    else:
        ddpg_class = DDPG
    algorithm = ddpg_class(
        env,
        es,
        policy,
        qf,
        batch_norm_config=bn_config,
        **variant['algo_params']
    )
    algorithm.train()


def my_ddpg_launcher(variant):
    """
    Run DDPG
    :param variant: Dictionary of dictionary with the following keys:
        - algo_params
        - env_params
        - qf_params
        - policy_params
    :return:
    """
    from railrl.algos.ddpg import DDPG
    from railrl.policies.nn_policy import FeedForwardPolicy
    from railrl.qfunctions.nn_qfunction import FeedForwardCritic
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from railrl.launchers.launcher_util import get_env_settings
    from railrl.core.tf_util import BatchNormConfig
    if ('batch_norm_params' in variant
        and variant['batch_norm_params'] is not None):
        bn_config = BatchNormConfig(**variant['batch_norm_params'])
    else:
        bn_config = None
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
        batch_norm_config=bn_config,
        **variant.get('qf_params', {})
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
        batch_norm_config=bn_config,
        **variant.get('policy_params', {})
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        batch_norm_config=bn_config,
        **variant['algo_params']
    )
    algorithm.train()


def quadratic_ddpg_launcher(variant):
    """
    Run DDPG with Quadratic Critic
    :param variant: Dictionary of dictionary with the following keys:
        - algo_params
        - env_params
        - qf_params
        - policy_params
    :return:
    """
    from railrl.algos.ddpg import DDPG as MyDDPG
    from railrl.policies.nn_policy import FeedForwardPolicy
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF
    from railrl.launchers.launcher_util import get_env_settings
    from railrl.core.tf_util import BatchNormConfig
    if ('batch_norm_params' in variant
        and variant['batch_norm_params'] is not None):
        bn_config = BatchNormConfig(**variant['batch_norm_params'])
    else:
        bn_config = None
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    es = OUStrategy(env_spec=env.spec)
    qf = QuadraticNAF(
        name_or_scope="critic",
        env_spec=env.spec,
        batch_norm_config=bn_config,
        **variant['qf_params']
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
        batch_norm_config=bn_config,
        **variant['policy_params']
    )
    algorithm = MyDDPG(
        env,
        es,
        policy,
        qf,
        batch_norm_config=bn_config,
        **variant['algo_params']
    )
    algorithm.train()


def oat_qddpg_launcher(variant):
    """
    Quadratic optimal action target DDPG
    """
    from railrl.algos.optimal_action_target_ddpg import \
        OptimalActionTargetDDPG as OAT
    from railrl.policies.nn_policy import FeedForwardPolicy
    from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from railrl.launchers.launcher_util import get_env_settings
    from railrl.core.tf_util import BatchNormConfig
    if ('batch_norm_params' in variant
        and variant['batch_norm_params'] is not None):
        bn_config = BatchNormConfig(**variant['batch_norm_params'])
    else:
        bn_config = None
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    es = OUStrategy(env_spec=env.spec)
    qf = QuadraticNAF(
        name_or_scope="critic",
        env_spec=env.spec,
        batch_norm_config=bn_config,
        **variant['qf_params']
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
        batch_norm_config=bn_config,
        **variant['policy_params']
    )
    algorithm = OAT(
        env,
        es,
        policy,
        qf,
        batch_norm_config=bn_config,
        **variant['algo_params']
    )
    algorithm.train()


def naf_launcher(variant):
    from railrl.algos.naf import NAF
    from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from railrl.launchers.launcher_util import get_env_settings
    from railrl.core.tf_util import BatchNormConfig
    if ('batch_norm_params' in variant
        and variant['batch_norm_params'] is not None):
        bn_config = BatchNormConfig(**variant['batch_norm_params'])
    else:
        bn_config = None
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    if 'es_init' in variant:
        es = variant['es_init'](env, **variant['exploration_strategy_params'])
    else:
        es = OUStrategy(
            env_spec=env.spec,
            **variant['exploration_strategy_params']
        )
    qf = QuadraticNAF(
        name_or_scope="qf",
        env_spec=env.spec,
        batch_norm_config=bn_config,
    )
    algorithm = NAF(
        env,
        es,
        qf,
        batch_norm_config=bn_config,
        **variant['algo_params']
    )
    algorithm.train()


def get_naf_ddpg_params():
    import tensorflow as tf
    # TODO: try this
    variant = {
        'Algorithm': 'NAF-DDPG',
        'quadratic_policy_params': dict(
            observation_hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
        ),
        'policy_params': dict(
            observation_hidden_sizes=(100, 100),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
        )
    }
    return variant


####################
# other algorithms #
####################
def shane_ddpg_launcher(variant):
    from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
    from sandbox.rocky.tf.algos.ddpg import DDPG as ShaneDDPG
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.tf.policies.deterministic_mlp_policy import (
        DeterministicMLPPolicy
    )
    from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import (
        ContinuousMLPQFunction
    )
    from railrl.launchers.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = TfEnv(env_settings['env'])
    es = GaussianStrategy(env.spec)

    policy = DeterministicMLPPolicy(
        name="init_policy",
        env_spec=env.spec,
        **variant['policy_params']
    )
    qf = ContinuousMLPQFunction(
        name="qf",
        env_spec=env.spec,
        **variant['qf_params']
    )

    algorithm = ShaneDDPG(
        env,
        policy,
        qf,
        es,
        **variant['algo_params']
    )
    algorithm.train()


def rllab_vpg_launcher(variant):
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from sandbox.rocky.tf.algos.vpg import VPG
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from railrl.launchers.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = TfEnv(env_settings['env'])
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algorithm = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        **variant['algo_params']
    )
    algorithm.train()


def rllab_trpo_launcher(variant):
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from sandbox.rocky.tf.algos.trpo import TRPO
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from railrl.launchers.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = TfEnv(env_settings['env'])
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algorithm = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        **variant['algo_params']
    )
    algorithm.train()


def rllab_ddpg_launcher(variant):
    from rllab.algos.ddpg import DDPG as RllabDDPG
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from rllab.q_functions.continuous_mlp_q_function import (
        ContinuousMLPQFunction as TheanoContinuousMLPQFunction
    )
    from rllab.policies.deterministic_mlp_policy import (
        DeterministicMLPPolicy as TheanoDeterministicMLPPolicy
    )
    from railrl.launchers.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    policy = TheanoDeterministicMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    es = OUStrategy(env_spec=env.spec)

    qf = TheanoContinuousMLPQFunction(env_spec=env.spec)

    algorithm = RllabDDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        **variant['algo_params']
    )
    algorithm.train()


def random_action_launcher(variant):
    from railrl.algos.noop_algo import NoOpAlgo
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from rllab.policies.uniform_control_policy import UniformControlPolicy
    from railrl.launchers.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    es = OUStrategy(env)
    policy = UniformControlPolicy(env_spec=env.spec)
    algorithm = NoOpAlgo(
        env,
        policy,
        es,
        **variant['algo_params']
    )
    algorithm.train()


def bptt_ddpg_launcher(variant):
    from railrl.algos.oracle_bptt_ddpg import OracleBpttDdpg
    from railrl.algos.meta_bptt_ddpg import MetaBpttDdpg
    from railrl.qfunctions.memory.oracle_unroll_qfunction import (
        OracleUnrollQFunction
    )
    from railrl.exploration_strategies.product_strategy import ProductStrategy
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )
    from railrl.policies.memory.lstm_memory_policy import LstmMemoryPolicy
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.qfunctions.memory.hint_mlp_memory_qfunction import (
        HintMlpMemoryQFunction
    )
    from os.path import exists
    import railrl.core.neuralnet
    railrl.core.neuralnet.dropout_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")

    """
    Set up experiment variants.
    """
    seed = variant['seed']
    load_policy_file = variant.get('load_policy_file', None)
    memory_dim = variant['memory_dim']
    oracle_mode = variant['oracle_mode']
    env_class = variant['env_class']
    env_params = variant['env_params']
    ddpg_params = variant['ddpg_params']
    policy_params = variant['policy_params']
    qf_params = variant['qf_params']
    meta_qf_params = variant['meta_qf_params']
    es_params = variant['es_params']
    replay_buffer_class = variant['replay_buffer_class']
    replay_buffer_params = variant['replay_buffer_params']
    replay_buffer_params['memory_dim'] = memory_dim

    env_es_class = es_params['env_es_class']
    env_es_params = es_params['env_es_params']
    memory_es_class = es_params['memory_es_class']
    memory_es_params = es_params['memory_es_params']
    noise_action_to_memory = es_params['noise_action_to_memory']
    num_bptt_unrolls = ddpg_params['num_bptt_unrolls']
    set_seed(seed)

    """
    Code for running the experiment.
    """

    ocm_env = env_class(**env_params)
    env_action_dim = ocm_env.action_space.flat_dim
    env_obs_dim = ocm_env.observation_space.flat_dim
    H = ocm_env.horizon
    env = ContinuousMemoryAugmented(
        ocm_env,
        num_memory_states=memory_dim,
    )

    policy = None
    qf = None
    if load_policy_file is not None and exists(load_policy_file):
        with tf.Session():
            data = joblib.load(load_policy_file)
            policy = data['policy']
            qf = data['qf']
    env_strategy = env_es_class(
        env_spec=ocm_env.spec,
        **env_es_params
    )
    write_strategy = memory_es_class(
        env_spec=env.memory_spec,
        **memory_es_params
    )
    if noise_action_to_memory:
        es = ActionAwareMemoryStrategy(
            env_strategy=env_strategy,
            write_strategy=write_strategy,
        )
        policy = policy or ActionAwareMemoryPolicy(
            name_or_scope="noisy_policy",
            action_dim=env_action_dim,
            memory_dim=memory_dim,
            env_spec=env.spec,
            **policy_params
        )
    else:
        es = ProductStrategy([env_strategy, write_strategy])
        policy = policy or LstmMemoryPolicy(
            name_or_scope="policy",
            action_dim=env_action_dim,
            memory_dim=memory_dim,
            env_spec=env.spec,
            num_env_obs_dims_to_use=env_obs_dim,
            **policy_params
        )

    ddpg_params = ddpg_params.copy()
    if oracle_mode == 'none':
        qf_params['use_time'] = False
        qf_params['use_target'] = False
        qf = HintMlpMemoryQFunction(
            name_or_scope="critic",
            hint_dim=env_action_dim,
            max_time=H,
            env_spec=env.spec,
            **qf_params
        )
        algo_class = variant['algo_class']
    elif oracle_mode == 'oracle' or oracle_mode == 'meta':
        oracle_params = variant['oracle_params']
        qf = qf or HintMlpMemoryQFunction(
            name_or_scope="hint_critic",
            hint_dim=env_action_dim,
            max_time=H,
            env_spec=env.spec,
            **qf_params
        )
        oracle_qf = OracleUnrollQFunction(
            name_or_scope="oracle_unroll_critic",
            env=env,
            policy=policy,
            num_bptt_unrolls=num_bptt_unrolls,
            env_obs_dim=env_obs_dim,
            env_action_dim=env_action_dim,
            max_horizon_length=H,
            env_spec=env.spec,
        )
        algo_class = OracleBpttDdpg
        ddpg_params['oracle_qf'] = oracle_qf
        ddpg_params.update(oracle_params)
    else:
        raise Exception("Unknown mode: {}".format(oracle_mode))
    if oracle_mode == 'meta':
        meta_qf = HintMlpMemoryQFunction(
            name_or_scope="meta_critic",
            hint_dim=env_action_dim,
            max_time=H,
            env_spec=env.spec,
            **meta_qf_params
        )
        algo_class = MetaBpttDdpg
        meta_params = variant['meta_params']
        ddpg_params['meta_qf'] = meta_qf
        ddpg_params.update(meta_params)

    algorithm = algo_class(
        env=env,
        exploration_strategy=es,
        policy=policy,
        qf=qf,
        env_obs_dim=env_obs_dim,
        env_action_dim=env_action_dim,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_params=replay_buffer_params,
        **ddpg_params
    )

    algorithm.train()
    return algorithm