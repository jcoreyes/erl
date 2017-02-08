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
