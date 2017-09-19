from railrl.torch.core import PyTorchModule
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
import tensorflow as tf
from rllab.misc import logger

filename = str(uuid.uuid4())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()

    policy = None
    env = None

    with tf.Session() as sess:
        import railrl.core.neuralnet
        railrl.core.neuralnet.dropout_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")
        data = joblib.load(args.file)
        if 'policy' in data:
            policy = data['policy']
        elif 'naf_policy' in data:
            policy = data['naf_policy']
        else:
            qf = data['optimizable_qfunction']
            policy = qf.implicit_policy
        env = data['env']
        print("Policy loaded")
        if args.gpu:
            set_gpu_mode(True)
            policy.cuda()
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
                    speedup=args.speedup,
                    always_return_paths=True,
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
