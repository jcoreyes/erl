"""
This file contains 'launchers', which are self-contained functions that take
one dictionary and run a full experiment. The dictionary configures the
experiment.

It is important that the functions are completely self-contained (i.e. they
import their own modules) so that they can be serialized.
"""
from rllab.misc.instrument import run_experiment_lite
import logging


#################
# my algorithms #
#################

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
    from algos.ddpg import DDPG as MyDDPG
    from policies.nn_policy import FeedForwardPolicy
    from qfunctions.nn_qfunction import FeedForwardCritic
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from misc.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
        **variant['qf_params']
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
        **variant['policy_params']
    )
    algorithm = MyDDPG(
        env,
        es,
        policy,
        qf,
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
    from algos.ddpg import DDPG as MyDDPG
    from policies.nn_policy import FeedForwardPolicy
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from qfunctions.quadratic_naf_qfunction import QuadraticNAF
    from misc.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    es = OUStrategy(env_spec=env.spec)
    qf = QuadraticNAF(
        name_or_scope="critic",
        env_spec=env.spec,
        **variant['qf_params']
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
        **variant['policy_params']
    )
    algorithm = MyDDPG(
        env,
        es,
        policy,
        qf,
        **variant['algo_params']
    )
    algorithm.train()


def oat_qddpg_launcher(variant):
    """
    Quadratic optimal action target DDPG
    """
    from algos.optimal_action_target_ddpg import OptimalActionTargetDDPG as OAT
    from policies.nn_policy import FeedForwardPolicy
    from qfunctions.quadratic_naf_qfunction import QuadraticNAF
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from misc.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    es = OUStrategy(env_spec=env.spec)
    qf = QuadraticNAF(
        name_or_scope="critic",
        env_spec=env.spec,
        **variant['qf_params']
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
        **variant['policy_params']
    )
    algorithm = OAT(
        env,
        es,
        policy,
        qf,
        **variant['algo_params']
    )
    algorithm.train()


def naf_launcher(variant):
    from algos.naf import NAF
    from qfunctions.quadratic_naf_qfunction import QuadraticNAF
    from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
    from misc.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    es = GaussianStrategy(env)
    qf = QuadraticNAF(
        name_or_scope="qf",
        env_spec=env.spec,
    )
    algorithm = NAF(
        env,
        es,
        qf,
        **variant['algo_params']
    )
    algorithm.train()


def naf_ddpg_launcher(variant):
    # TODO: try this
    """Basically implement NAF with the DDPG code. The only difference is that
    the actor now gets updated twice."""
    from algos.ddpg import DDPG as MyDDPG
    from policies.nn_policy import FeedForwardPolicy
    from qfunctions.quadratic_qf import QuadraticQF
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from misc.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    es = OUStrategy(env_spec=env.spec)
    quadratic_policy = FeedForwardPolicy(
        name_or_scope="quadratic_policy",
        env_spec=env.spec,
        **variant['quadratic_policy_params']
    )
    qf = QuadraticQF(
        name_or_scope="quadratic_qfunction",
        env_spec=env.spec,
        policy=quadratic_policy,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
        **variant['policy_params']
    )
    algorithm = MyDDPG(
        env,
        es,
        policy,
        qf,
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


def convex_naf_launcher(variant):
    from algos.convex_naf import ConvexNAFAlgorithm
    from qfunctions.convex_naf_qfunction import ConcaveNAF
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from misc.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    # The ICNN paper uses the OU strategy
    # es = GaussianStrategy(env)
    es = OUStrategy(env_spec=env.spec, sigma=0.1)
    optimizer_type = variant['optimizer_type']
    qf = ConcaveNAF(
        name_or_scope="qf",
        env_spec=env.spec,
        optimizer_type=optimizer_type,
    )
    algorithm = ConvexNAFAlgorithm(
        env,
        es,
        qf,
        **variant['algo_params']
    )
    algorithm.train()


def convex_quadratic_naf_launcher(variant):
    from algos.convex_naf import ConvexNAFAlgorithm
    from qfunctions.sgd_quadratic_naf_qfunction import SgdQuadraticNAF
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from misc.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    optimizer_type = variant['optimizer_type']
    es = OUStrategy(env_spec=env.spec, sigma=0.1)
    qf = SgdQuadraticNAF(
        name_or_scope="qf",
        env_spec=env.spec,
    )
    algorithm = ConvexNAFAlgorithm(
        env,
        es,
        qf,
        **variant['algo_params']
    )
    algorithm.train()


def dqicnn_launcher(variant):
    from algos.dqicnn import DQICNN
    from qfunctions.action_concave_qfunction import ActionConcaveQFunction
    from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
    from misc.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    es = GaussianStrategy(env)
    qf = ActionConcaveQFunction(
        name_or_scope="qf",
        env_spec=env.spec,
    )
    algorithm = DQICNN(
        env,
        es,
        qf,
        **variant['algo_params']
    )
    algorithm.train()


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
    from misc.launcher_util import get_env_settings
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
    from misc.launcher_util import get_env_settings
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
    from misc.launcher_util import get_env_settings
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
    from misc.launcher_util import get_env_settings
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
    from algos.noop_algo import NoOpAlgo
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from rllab.policies.uniform_control_policy import UniformControlPolicy
    from misc.launcher_util import get_env_settings
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


def run_experiment(task, exp_prefix, seed, variant):
    variant['seed'] = str(seed)
    logger = logging.getLogger(__name__)
    logger.info("Variant:")
    logger.info(variant)
    run_experiment_lite(
        task,
        snapshot_mode="last",
        exp_prefix=exp_prefix,
        variant=variant,
        seed=seed,
        use_cloudpickle=True,
    )
