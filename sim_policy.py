from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
import tensorflow as tf

filename = str(uuid.uuid4())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    args = parser.parse_args()

    policy = None
    env = None

    with tf.Session() as sess:
        data = joblib.load(args.file)
        if '_policy' in data:
            policy = data['_policy']
        else:
            qf = data['optimizable_qfunction']
            policy = qf.implicit_policy
        env = data['env']
        while True:
            try:
                path = rollout(env, policy, max_path_length=args.max_path_length,
                               animated=True, speedup=args.speedup)
            # Hack for now. Not sure why rollout assumes that close is an
            # keyword argument
            except TypeError as e:
                if (str(e) != "render() got an unexpected keyword "
                              "argument 'close'"):
                    raise e
