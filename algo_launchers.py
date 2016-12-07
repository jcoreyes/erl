import tensorflow as tf

from algos.convex_naf import ConvexNAFAlgorithm
from algos.ddpg import DDPG as MyDDPG
from algos.naf import NAF
from algos.noop_algo import NoOpAlgo
from policies.nn_policy import FeedForwardPolicy
from qfunctions.convex_naf_qfunction import ConvexNAF
from qfunctions.nn_qfunction import FeedForwardCritic
from qfunctions.quadratic_naf_qfunction import QuadraticNAF
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from rllab.policies.uniform_control_policy import UniformControlPolicy
from sandbox.rocky.tf.algos.ddpg import DDPG as ShaneDDPG
from sandbox.rocky.tf.policies.deterministic_mlp_policy import \
    DeterministicMLPPolicy
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import \
    ContinuousMLPQFunction


def test_my_ddpg(env, exp_prefix, env_name, seed=1, **ddpg_params):
    es = OUStrategy(env_spec=env.spec)
    qf_params = dict(
        embedded_hidden_sizes=(100,),
        observation_hidden_sizes=(100,),
        # hidden_W_init=util.xavier_uniform_initializer,
        # hidden_b_init=tf.zeros_initializer,
        # output_W_init=util.xavier_uniform_initializer,
        # output_b_init=tf.zeros_initializer,
        hidden_nonlinearity=tf.nn.relu,
    )
    policy_params = dict(
        observation_hidden_sizes=(100, 100),
        # hidden_W_init=util.xavier_uniform_initializer,
        # hidden_b_init=tf.zeros_initializer,
        # output_W_init=util.xavier_uniform_initializer,
        # output_b_init=tf.zeros_initializer,
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    )
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
        **qf_params
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
        **policy_params
    )
    algorithm = MyDDPG(
        env,
        es,
        policy,
        qf,
        **ddpg_params
    )
    variant = ddpg_params
    variant['Version'] = 'Mine'
    variant['Environment'] = env_name
    variant['Algo'] = 'DDPG'
    for qf_key, qf_value in qf_params.items():
        variant['qf_' + qf_key] = str(qf_value)
    for policy_key, policy_value in policy_params.items():
        variant['policy_' + policy_key] = str(policy_value)
    run_experiment(algorithm, exp_prefix, seed, variant)


def test_ddpg_quadratic(env, exp_prefix, env_name, seed=1, **ddpg_params):
    es = OUStrategy(env_spec=env.spec)
    qf_params = dict(
        embedded_hidden_sizes=(100,),
        observation_hidden_sizes=(100,),
        # hidden_W_init=util.xavier_uniform_initializer,
        # hidden_b_init=tf.zeros_initializer,
        # output_W_init=util.xavier_uniform_initializer,
        # output_b_init=tf.zeros_initializer,
        hidden_nonlinearity=tf.nn.relu,
    )
    policy_params = dict(
        observation_hidden_sizes=(100, 100),
        # hidden_W_init=util.xavier_uniform_initializer,
        # hidden_b_init=tf.zeros_initializer,
        # output_W_init=util.xavier_uniform_initializer,
        # output_b_init=tf.zeros_initializer,
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    )
    quadratic_policy = FeedForwardPolicy(
        name_or_scope="quadratic_policy",
        env_spec=env.spec,
        observation_input=None,
        observation_hidden_sizes=(200, 200),
        hidden_W_init=None,
        hidden_b_init=None,
        output_W_init=None,
        output_b_init=None,
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    )
    qf = QuadraticQF(
        name_or_scope="quadratic_qfunction",
        action_input=None,
        observation_input=policy.observation_input,
        action_dim=self.action_dim,
        observation_dim=self.observation_dim,
        policy=self.policy,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
        **policy_params
    )
    algorithm = MyDDPG(
        env,
        es,
        policy,
        qf,
        **ddpg_params
    )
    variant = ddpg_params
    variant['Version'] = 'Mine'
    variant['Environment'] = env_name
    variant['Algo'] = 'QuadraticDDPG'
    for qf_key, qf_value in qf_params.items():
        variant['qf_' + qf_key] = str(qf_value)
    for policy_key, policy_value in policy_params.items():
        variant['policy_' + policy_key] = str(policy_value)
    run_experiment(algorithm, exp_prefix, seed, variant)

def test_my_naf(env, exp_prefix, env_name, seed=1, **naf_params):
    es = GaussianStrategy(env)
    qf = QuadraticNAF(
        name_or_scope="qf",
        env_spec=env.spec,
    )
    algorithm = NAF(
        env,
        es,
        qf,
        **naf_params
    )
    variant = naf_params
    variant['Version'] = 'Mine'
    variant['Environment'] = env_name
    variant['Algo'] = 'NAF'
    run_experiment(algorithm, exp_prefix, seed, variant)


def test_convex_naf(env, exp_prefix, env_name, seed=1, **naf_params):
    es = GaussianStrategy(env)
    qf = ConvexNAF(
        name_or_scope="qf",
        env_spec=env.spec,
    )
    algorithm = ConvexNAFAlgorithm(
        env,
        es,
        qf,
        **naf_params
    )
    variant = naf_params
    variant['Version'] = 'Mine'
    variant['Environment'] = env_name
    variant['Algo'] = 'ConvexNAF'
    run_experiment(algorithm, exp_prefix, seed, variant)

def test_shane_ddpg(env, exp_prefix, env_name, seed=1, **new_ddpg_params):
    ddpg_params = dict(get_ddpg_params(), **new_ddpg_params)
    es = GaussianStrategy(env.spec)

    policy_params = dict(
        hidden_sizes=(100, 100),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    )
    qf_params = dict(
        hidden_sizes=(100, 100)
    )
    policy = DeterministicMLPPolicy(
        name="init_policy",
        env_spec=env.spec,
        **policy_params
    )
    qf = ContinuousMLPQFunction(
        name="qf",
        env_spec=env.spec,
        **qf_params
    )

    algorithm = ShaneDDPG(
        env,
        policy,
        qf,
        es,
        **ddpg_params
    )

    variant = ddpg_params
    variant['Version'] = 'Shane'
    variant['Environment'] = env_name
    for qf_key, qf_value in qf_params.items():
        variant['qf_' + qf_key] = str(qf_value)
    for policy_key, policy_value in policy_params.items():
        variant['policy_' + policy_key] = str(policy_value)

    run_experiment(algorithm, exp_prefix, seed, variant=variant)


def test_random_ddpg(env, exp_prefix, env_name, seed=1, **algo_params):
    es = OUStrategy(env)
    policy = UniformControlPolicy(env_spec=env.spec)
    algorithm = NoOpAlgo(
        env,
        policy,
        es,
        **algo_params)
    variant = {'Version': 'Random', 'Environment': env_name}

    run_experiment(algorithm, exp_prefix, seed, variant=variant)


def run_experiment(algorithm, exp_prefix, seed, variant):
    variant['seed'] = str(seed)
    print("variant=")
    print(variant)
    run_experiment_lite(
        algorithm.train(),
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix=exp_prefix,
        variant=variant,
        seed=seed,
    )
stub(globals())
