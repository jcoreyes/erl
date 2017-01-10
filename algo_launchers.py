import tensorflow as tf

from algos.convex_naf import ConvexNAFAlgorithm
from algos.ddpg import DDPG as MyDDPG
from algos.dqicnn import DQICNN
from algos.naf import NAF
from qfunctions.action_concave_qfunction import ActionConcaveQFunction
from qfunctions.sgd_quadratic_naf_qfunction import SgdQuadraticNAF
from rllab.algos.ddpg import DDPG as RllabDDPG
from algos.noop_algo import NoOpAlgo
from policies.nn_policy import FeedForwardPolicy
from qfunctions.convex_naf_qfunction import ConcaveNAF
from qfunctions.nn_qfunction import FeedForwardCritic
from qfunctions.quadratic_naf_qfunction import QuadraticNAF
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from rllab.policies.uniform_control_policy import UniformControlPolicy
from sandbox.rocky.tf.algos.ddpg import DDPG as ShaneDDPG
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.deterministic_mlp_policy import (
    DeterministicMLPPolicy
)
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import (
    ContinuousMLPQFunction
)
from rllab.q_functions.continuous_mlp_q_function import (
    ContinuousMLPQFunction as TheanoContinuousMLPQFunction
)
from rllab.policies.deterministic_mlp_policy import (
    DeterministicMLPPolicy as TheanoDeterministicMLPPolicy
)


#################
# my algorithms #
#################

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
    variant['Environment'] = env_name
    variant['Algorithm'] = 'DDPG'
    for qf_key, qf_value in qf_params.items():
        variant['qf_' + qf_key] = str(qf_value)
    for policy_key, policy_value in policy_params.items():
        variant['policy_' + policy_key] = str(policy_value)
    run_experiment(algorithm, exp_prefix, seed, variant)


def test_ddpg_quadratic(env, exp_prefix, env_name, seed=1, **algo_params):
    es = OUStrategy(env_spec=env.spec)
    policy_params = dict(
        observation_hidden_sizes=(100, 100),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    )
    policy = FeedForwardPolicy(
        name_or_scope="policy",
        env_spec=env.spec,
        **policy_params
    )
    qf = QuadraticNAF(
        name_or_scope="quadratic_qf",
        env_spec=env.spec,
    )
    algorithm = MyDDPG(
        env,
        es,
        policy,
        qf,
        **algo_params
    )
    variant = algo_params
    variant['Version'] = 'Mine'
    variant['Environment'] = env_name
    variant['Algorithm'] = 'QuadraticDDPG'
    for policy_key, policy_value in policy_params.items():
        variant['policy_' + policy_key] = str(policy_value)
    run_experiment(algorithm, exp_prefix, seed, variant)


def test_naf_ddpg(env, exp_prefix, env_name, seed=1, **algo_params):
    """Basically implement NAF with the DDPG code. The only difference is that
    the actor now gets updated twice."""
    # TODO
    es = OUStrategy(env_spec=env.spec)
    quadratic_policy_params = dict(
        observation_hidden_sizes=(100, 100),
        hidden_W_init=None,
        hidden_b_init=None,
        output_W_init=None,
        output_b_init=None,
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    )
    policy_params = dict(
        observation_hidden_sizes=(100, 100),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    )
    quadratic_policy = FeedForwardPolicy(
        name_or_scope="quadratic_policy",
        env_spec=env.spec,
        **quadratic_policy_params
    )
    qf = QuadraticQF(
        name_or_scope="quadratic_qfunction",
        env_spec=env.spec,
        policy=quadratic_policy,
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
        **algo_params
    )
    variant = algo_params
    variant['Version'] = 'Mine'
    variant['Environment'] = env_name
    variant['Algo'] = 'QuadraticDDPG'
    for qf_key, qf_value in quadratic_policy_params.items():
        variant['quadratic_policy_params_' + qf_key] = str(qf_value)
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
    variant['Environment'] = env_name
    variant['Algorithm'] = 'NAF'
    run_experiment(algorithm, exp_prefix, seed, variant)


def test_convex_naf(env, exp_prefix, env_name, seed=1, **naf_params):
    # The ICNN paper uses the OU strategy
    # es = GaussianStrategy(env)
    es = OUStrategy(env_spec=env.spec, sigma=0.1)
    optimizer_type = naf_params.pop('optimizer_type', 'sgd')
    qf = ConcaveNAF(
        name_or_scope="qf",
        env_spec=env.spec,
        optimizer_type=optimizer_type,
    )
    algorithm = ConvexNAFAlgorithm(
        env,
        es,
        qf,
        **naf_params
    )
    variant = naf_params
    variant['optimizer_type'] = optimizer_type
    variant['Environment'] = env_name
    variant['Algorithm'] = 'ConvexNAF'
    run_experiment(algorithm, exp_prefix, seed, variant)


def test_convex_quadratic_naf(env, exp_prefix, env_name, seed=1, **naf_params):
    # The ICNN paper uses the OU strategy
    # es = GaussianStrategy(env)
    optimizer_type = naf_params.pop('optimizer_type', 'sgd')
    es = OUStrategy(env_spec=env.spec, sigma=0.1)
    qf = SgdQuadraticNAF(
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
    variant['Environment'] = env_name
    variant['Algorithm'] = 'ConvexQuadraticNAF'
    run_experiment(algorithm, exp_prefix, seed, variant)

def test_dqicnn(env, exp_prefix, env_name, seed=1, **naf_params):
    es = GaussianStrategy(env)
    qf = ActionConcaveQFunction(
        name_or_scope="qf",
        env_spec=env.spec,
    )
    algorithm = DQICNN(
        env,
        es,
        qf,
        **naf_params
    )
    variant = naf_params
    variant['Environment'] = env_name
    variant['Algorithm'] = 'DQICNN'
    run_experiment(algorithm, exp_prefix, seed, variant)


####################
# other algorithms #
####################
def test_shane_ddpg(env_, exp_prefix, env_name, seed=1, **ddpg_params):
    env = TfEnv(env_)
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
    variant['Algorithm'] = 'Shane-DDPG'
    variant['Environment'] = env_name
    for qf_key, qf_value in qf_params.items():
        variant['qf_' + qf_key] = str(qf_value)
    for policy_key, policy_value in policy_params.items():
        variant['policy_' + policy_key] = str(policy_value)

    run_experiment(algorithm, exp_prefix, seed, variant=variant)


def test_rllab_vpg(env_, exp_prefix, env_name, seed=1, **algo_params):
    env = TfEnv(env_)
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
        **algo_params
    )
    variant = algo_params
    variant['Environment'] = env_name
    variant['Algorithm'] = 'rllab-VPG'
    run_experiment(algorithm, exp_prefix, seed, variant=variant)


def test_rllab_trpo(env_, exp_prefix, env_name, seed=1, **algo_params):
    env = TfEnv(env_)
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
        **algo_params
    )
    variant = algo_params
    variant['Environment'] = env_name
    variant['Algorithm'] = 'rllab-TRPO'
    run_experiment(algorithm, exp_prefix, seed, variant=variant)


def test_rllab_ddpg(env, exp_prefix, env_name, seed=1, **algo_params):
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
        **algo_params
    )
    variant = algo_params
    variant['Environment'] = env_name
    variant['Algorithm'] = 'rllab-DDPG'
    run_experiment(algorithm, exp_prefix, seed, variant=variant)


def test_random(env, exp_prefix, env_name, seed=1, **algo_params):
    es = OUStrategy(env)
    policy = UniformControlPolicy(env_spec=env.spec)
    algorithm = NoOpAlgo(
        env,
        policy,
        es,
        **algo_params)
    variant = algo_params
    variant['Environment'] = env_name
    variant['Algorithm'] = 'Random'

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
