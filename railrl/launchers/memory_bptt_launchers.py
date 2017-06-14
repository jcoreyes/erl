"""
Each launcher variant should contain the following keys:
    - H
    - num_steps_per_iteration
    - num_steps_per_eval 
    - num_iterations
    - env_class
    - env_params
    - seed
    - memory_dim
    - use_gpu
    - batch_size
"""


def update_variant(old_variant, variants_to_add):
    from rllab.misc import logger
    import os.path as osp
    from railrl.pythonplusplus import merge_recursive_dicts
    import copy
    new_variant = copy.deepcopy(old_variant)
    merge_recursive_dicts(new_variant, variants_to_add)
    log_dir = logger.get_snapshot_dir()
    logger.log_variant(
        osp.join(log_dir, "variant.json"),
        new_variant
    )
    return new_variant


def super_trpo_launcher(variant, sub_launcher, version):
    num_steps_per_iteration = variant['num_steps_per_iteration']
    H = variant['H']
    num_iterations = variant['num_iterations']
    new_variant = update_variant(
        variant,
        dict(
            trpo_params=dict(
                batch_size=num_steps_per_iteration,
                max_path_length=H,  # Environment should stop it
                n_itr=num_iterations,
                discount=1.,
                step_size=0.01,
            ),
            optimizer_params=dict(
                base_eps=1e-5,
            ),
            version=version,
        ),
    )
    sub_launcher(new_variant)


def trpo_launcher(variant):
    super_trpo_launcher(variant, _trpo_launcher, "TRPO")


def rtrpo_launcher(variant):
    super_trpo_launcher(variant, _rtrpo_launcher, "Recurrent TRPO")


def mem_trpo_launcher(variant):
    super_trpo_launcher(variant, _mem_trpo_launcher, "Memory States + TRPO")


def _trpo_launcher(variant):
    from sandbox.rocky.tf.algos.trpo import TRPO
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import (
        ConjugateGradientOptimizer,
        FiniteDifferenceHvp,
    )
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.envs.wrappers import convert_to_tf_env
    env_class = variant['env_class']
    env_params = variant['env_params']
    seed = variant['seed']
    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = env_class(**env_params)
    env = convert_to_tf_env(env)

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(32, 32),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    optimizer_params = variant['optimizer_params']
    trpo_params = variant['trpo_params']
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(
            **optimizer_params
        )),
        **trpo_params
    )

    algo.train()


def _rtrpo_launcher(variant):
    from sandbox.rocky.tf.algos.trpo import TRPO
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
    import sandbox.rocky.tf.core.layers as L
    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import (
        ConjugateGradientOptimizer,
        FiniteDifferenceHvp,
    )
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.envs.wrappers import convert_to_tf_env
    env_class = variant['env_class']
    env_params = variant['env_params']
    seed = variant['seed']
    set_seed(seed)
    env = env_class(**env_params)
    env = convert_to_tf_env(env)

    policy = GaussianLSTMPolicy(
        name="policy",
        env_spec=env.spec,
        lstm_layer_cls=L.LSTMLayer,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    optimizer_params = variant['optimizer_params']
    trpo_params = variant['trpo_params']
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(
            **optimizer_params
        )),
        **trpo_params
    )

    algo.train()


def _mem_trpo_launcher(variant):
    from sandbox.rocky.tf.algos.trpo import TRPO
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import (
        ConjugateGradientOptimizer,
        FiniteDifferenceHvp,
    )
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.envs.flattened_product_box import FlattenedProductBox
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )

    """
    Set up experiment variants.
    """
    seed = variant['seed']
    env_class = variant['env_class']
    env_params = variant['env_params']
    memory_dim = variant['memory_dim']

    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = env_class(**env_params)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=memory_dim,
    )
    env = FlattenedProductBox(env)

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(32, 32),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    optimizer_params = variant['optimizer_params']
    trpo_params = variant['trpo_params']
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(
            **optimizer_params
        )),
        **trpo_params
    )

    algo.train()


def super_ddpg_launcher(variant, sub_launcher, version):
    H = variant['H']
    num_steps_per_iteration = variant['num_steps_per_iteration']
    num_steps_per_eval = variant['num_steps_per_eval']
    num_iterations = variant['num_iterations']
    use_gpu = variant['use_gpu']
    batch_size = variant['batch_size']
    new_variant = update_variant(
        variant,
        dict(
            algo_params=dict(
                batch_size=batch_size,
                num_epochs=num_iterations,
                pool_size=1000000,
                num_steps_per_epoch=num_steps_per_iteration,
                num_steps_per_eval=num_steps_per_eval,
                max_path_length=H,
                discount=1,
                use_gpu=use_gpu,
            ),
            ou_params=dict(
                max_sigma=1,
                min_sigma=None,
            ),
            version=version,
        )
    )
    sub_launcher(new_variant)


def mem_ddpg_launcher(variant):
    super_ddpg_launcher(variant, _mem_ddpg_launcher, "Memory States + DDPG")


def ddpg_launcher(variant):
    super_ddpg_launcher(variant, _ddpg_launcher, "DDPG")


def rdpg_launcher(variant):
    super_ddpg_launcher(variant, _rdpg_launcher, "Recurrent DPG")


def _mem_ddpg_launcher(variant):
    from railrl.torch.ddpg import DDPG
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.exploration_strategies.ou_strategy import OUStrategy
    from railrl.policies.torch import FeedForwardPolicy
    from railrl.qfunctions.torch import FeedForwardQFunction
    from railrl.envs.flattened_product_box import FlattenedProductBox
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )

    """
    Set up experiment variants.
    """
    seed = variant['seed']
    algo_params = variant['algo_params']
    env_class = variant['env_class']
    env_params = variant['env_params']
    ou_params = variant['ou_params']
    memory_dim = variant['memory_dim']

    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = env_class(**env_params)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=memory_dim,
    )
    env = FlattenedProductBox(env)

    es = OUStrategy(
        action_space=env.action_space,
        **ou_params
    )
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    algorithm = DDPG(
        env,
        es,
        qf=qf,
        policy=policy,
        **algo_params
    )

    algorithm.train()


def _ddpg_launcher(variant):
    from railrl.torch.ddpg import DDPG
    from railrl.launchers.launcher_util import set_seed
    from railrl.exploration_strategies.ou_strategy import OUStrategy
    from railrl.policies.torch import FeedForwardPolicy
    from railrl.qfunctions.torch import FeedForwardQFunction

    """
    Set up experiment variants.
    """
    seed = variant['seed']
    algo_params = variant['algo_params']
    env_class = variant['env_class']
    env_params = variant['env_params']
    ou_params = variant['ou_params']

    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = env_class(**env_params)

    es = OUStrategy(
        action_space=env.action_space,
        **ou_params
    )
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    algorithm = DDPG(
        env,
        es,
        qf=qf,
        policy=policy,
        **algo_params
    )

    algorithm.train()


def _rdpg_launcher(variant):
    from railrl.exploration_strategies.ou_strategy import OUStrategy
    from railrl.launchers.launcher_util import set_seed
    from railrl.policies.torch import RecurrentPolicy
    from railrl.qfunctions.torch import RecurrentQFunction
    from railrl.torch.rdpg import Rdpg
    seed = variant['seed']
    set_seed(seed)
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    es = OUStrategy(action_space=env.action_space)
    qf = RecurrentQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        10,
    )
    policy = RecurrentPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        10,
    )
    algorithm = Rdpg(
        env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        **variant['algo_params']
    )
    algorithm.train()
