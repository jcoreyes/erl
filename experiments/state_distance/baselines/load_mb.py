import tensorflow as tf

from controllers_solution import MPCcontroller
from dynamics_solution import NNDynamicsModel
from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger

filename = str(uuid.uuid4())


def simulate_policy(args):
    dir = args.path
    data = joblib.load("{}/params.pkl".format(dir))
    env = data['env']
    model_params = data['model_params']
    mpc_params = data['mpc_params']
    # dyn_model = NNDynamicsModel(env=env, **model_params)
    # mpc_controller = MPCcontroller(env=env,
    #                                dyn_model=dyn_model,
    #                                **mpc_params)
    tf_path_meta = "{}/tf_out-0.meta".format(dir)
    tf_path = "{}/tf_out-0".format(dir)

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(tf_path_meta)
        new_saver.restore(sess, tf_path)

    env = data['env']
    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.to(ptu.device)
    if args.pause:
        import ipdb; ipdb.set_trace()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    while True:
        try:
            path = rollout(
                env,
                policy,
                max_path_length=args.H,
                animated=True,
            )
            env.log_diagnostics([path])
            policy.log_diagnostics([path])
            logger.dump_tabular()
        # Hack for now. Not sure why rollout assumes that close is an
        # keyword argument
        except TypeError as e:
            if (str(e) != "render() got an unexpected keyword "
                          "argument 'close'"):
                raise e

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)

    # TODO(vitchyr): maybe add this check back in with a try-except statement
    # import tensorflow as tf
    # with tf.Session() as sess:
    #     simulate_policy(args)
