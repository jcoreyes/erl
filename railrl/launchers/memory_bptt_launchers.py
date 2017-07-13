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

There's a bunch of launchers that create new_variants. These sub-launchers
are basically adding their own default values to the variant passed in.

The launchers that finally call `algo.train()` are the final launchers.
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
    memory_dim = variant['memory_dim']
    set_seed(seed)
    env = env_class(**env_params)
    env = convert_to_tf_env(env)

    policy = GaussianLSTMPolicy(
        name="policy",
        env_spec=env.spec,
        lstm_layer_cls=L.LSTMLayer,
        hidden_dim=memory_dim,
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
        env=env,
        exploration_strategy=es,
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
        env=env,
        qf=qf,
        policy=policy,
        exploration_strategy=es,
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
    memory_dim = variant['memory_dim']
    es = OUStrategy(
        action_space=env.action_space,
        **variant['ou_params']
    )
    qf = RecurrentQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        hidden_size=100,
        fc1_size=100,
        fc2_size=100,
    )
    policy = RecurrentPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        memory_dim,
        fc1_size=100,
        fc2_size=100,
    )
    algorithm = Rdpg(
        env,
        qf,
        policy,
        es,
        **variant['algo_params']
    )
    algorithm.train()


def our_method_launcher(variant):
    from railrl.pythonplusplus import identity
    from railrl.torch.rnn import GRUCell, BNLSTMCell
    from railrl.policies.torch import RWACell
    from railrl.qfunctions.torch import MemoryQFunction
    from railrl.exploration_strategies.ou_strategy import OUStrategy
    from railrl.envs.memory.high_low import HighLow
    from torch.nn import functional as F
    H = variant['H']
    num_steps_per_iteration = variant['num_steps_per_iteration']
    num_steps_per_eval = variant['num_steps_per_eval']
    num_iterations = variant['num_iterations']
    batch_size = variant['batch_size']
    version = "Our Method"
    assert variant['env_class'] == HighLow, "cell_class hardcoded for HighLow"
    # noinspection PyTypeChecker
    new_variant = update_variant(
        variant,
        dict(
            memory_aug_params=dict(
                max_magnitude=1,
            ),
            algo_params=dict(
                subtraj_length=H,
                batch_size=batch_size,
                num_epochs=num_iterations,
                num_steps_per_epoch=num_steps_per_iteration,
                num_steps_per_eval=num_steps_per_eval,
                discount=0.9,
                use_action_policy_params_for_entire_policy=False,
                action_policy_optimize_bellman=False,
                write_policy_optimizes='bellman',
                action_policy_learning_rate=0.000980014225523977,
                write_policy_learning_rate=0.0005,
                qf_learning_rate=0.002021863834563243,
                max_path_length=H,
                refresh_entire_buffer_period=None,
                save_new_memories_back_to_replay_buffer=True,
                write_policy_weight_decay=0,
                action_policy_weight_decay=0,
            ),
            # qf_class=RecurrentMemoryQFunction,
            qf_class=MemoryQFunction,
            qf_params=dict(
                output_activation=identity,
                # hidden_size=10,
                fc1_size=400,
                fc2_size=300,
            ),
            policy_params=dict(
                fc1_size=400,
                fc2_size=300,
                # cell_class=GRUCell,
                # cell_class=BNLSTMCell,
                cell_class=RWACell,
                output_activation=F.tanh,
            ),
            es_params=dict(
                env_es_class=OUStrategy,
                env_es_params=dict(
                    max_sigma=1,
                    min_sigma=None,
                ),
                memory_es_class=OUStrategy,
                memory_es_params=dict(
                    max_sigma=1,
                    min_sigma=None,
                ),
            ),
            version=version,
        )
    )
    _our_method_launcher(new_variant)


def _our_method_launcher(variant):
    from railrl.torch.bptt_ddpg import BpttDdpg
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.exploration_strategies.product_strategy import ProductStrategy
    from railrl.policies.torch import MemoryPolicy
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )
    seed = variant['seed']
    algo_params = variant['algo_params']
    memory_dim = variant['memory_dim']
    rnn_cell = variant['policy_params']['cell_class']
    memory_dim -= memory_dim % rnn_cell.state_num_split()
    env_class = variant['env_class']
    env_params = variant['env_params']
    memory_aug_params = variant['memory_aug_params']

    qf_class = variant['qf_class']
    qf_params = variant['qf_params']
    policy_params = variant['policy_params']

    es_params = variant['es_params']
    env_es_class = es_params['env_es_class']
    env_es_params = es_params['env_es_params']
    memory_es_class = es_params['memory_es_class']
    memory_es_params = es_params['memory_es_params']

    set_seed(seed)
    raw_env = env_class(**env_params)
    env = ContinuousMemoryAugmented(
        raw_env,
        num_memory_states=memory_dim,
        **memory_aug_params
    )
    env_strategy = env_es_class(
        action_space=raw_env.action_space,
        **env_es_params
    )
    write_strategy = memory_es_class(
        action_space=env.memory_state_space,
        **memory_es_params
    )
    es = ProductStrategy([env_strategy, write_strategy])
    qf = qf_class(
        int(raw_env.observation_space.flat_dim),
        int(raw_env.action_space.flat_dim),
        memory_dim,
        **qf_params,
    )
    policy = MemoryPolicy(
        int(raw_env.observation_space.flat_dim),
        int(raw_env.action_space.flat_dim),
        memory_dim=memory_dim,
        **policy_params
    )
    algorithm = BpttDdpg(
        env,
        qf,
        policy,
        es,
        **algo_params
    )
    algorithm.train()
