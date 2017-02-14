import argparse
import joblib
import uuid
import numpy as np
import tensorflow as tf

from railrl.misc.np_util import np_print_options

filename = str(uuid.uuid4())

def print_episode(episode, target):
    """
    :param episode: List of actions
    :return:
    """
    with np_print_options(precision=3, suppress=True):
        for t, action in enumerate(episode):
            print("target={0}\tt={1}\taction={2}".format(
                target,
                t,
                action,
            ))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    with tf.Session() as sess:
        with sess.as_default():
            data = joblib.load(args.file)
            bptt = data['bptt']
            env = bptt._env
            while True:
                X, Y = env.get_batch(bptt._batch_size)
                import ipdb
                ipdb.set_trace()
                # convert one hot to number
                targets = [np.nonzero(episode[0])[0][0] for episode in X]

                predictions = bptt.get_prediction(X)
                episode_length = len(predictions)
                episodes = []
                for i in range(bptt._batch_size):
                    actions = []
                    for t in range(episode_length):
                        actions.append(predictions[t][i, :])
                    episodes.append(actions)

                assert len(episodes) == len(targets)
                for episode, target in zip(episodes, targets):
                    print_episode(episode, target)
