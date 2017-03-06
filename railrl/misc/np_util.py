import numpy as np
import contextlib


@contextlib.contextmanager
def np_print_options(*args, **kwargs):
    """
    Locally modify print behavior.

    Usage:
    ```
    x = np.random.random(10)
    with printoptions(precision=3, suppress=True):
        print(x)
        # [ 0.073  0.461  0.689  0.754  0.624  0.901  0.049  0.582  0.557  0.348]
    ```

    http://stackoverflow.com/questions/2891790/how-to-pretty-printing-a-numpy-array-without-scientific-notation-and-with-given
    :param args:
    :param kwargs:
    :return:
    """
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


# TODO(vpong): Test this
def to_onehot(x, num_values):
    """
    Return a one hot vector representing x.
    :param x: Number to represent.
    :param num_values: Size of onehot vector.
    :return: nd.array of shape (num_values,)
    """
    onehot = np.zeros(num_values)
    onehot[x] = 1
    return onehot


def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    denom = np.expand_dims(e_x.sum(axis=axis), axis=axis)
    return e_x / denom


def subsequences(tensor, start_indices, length, start_offset=0):
    """
    Return subsequences of a tensor, starting at the indices give by
    `start_indices` plus `start_offset`.
    :param tensor: np.array
        Shape: n x m1 x m2 x ... x md
        where *m could be a number, or
    :param start_indices: list, length k
    :param length: int
    :param start_offset: int
    :return: np.array
        shape: k x length x m1 x m2 x ... md
    """
    tensor_shape = tensor.shape
    if len(tensor_shape) == 1:
        result = np.empty((len(start_indices), length), dtype=tensor.dtype)
        for i, start_i in enumerate(start_indices):
            result[i] = tensor[start_i+start_offset:start_i+start_offset+length]
    else:
        result = np.empty((len(start_indices), length) + tensor_shape[1:],
                          dtype=tensor.dtype)
        for i, start_i in enumerate(start_indices):
            result[i, :] = (
                tensor[start_i+start_offset:start_i+start_offset+length, :]
            )
    return result
