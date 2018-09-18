import numpy as np


def gaussian_data(batch_size):
    return np.random.randn(batch_size, 2)


def small_gaussian_data(batch_size):
    return 0.5*np.random.randn(batch_size, 2)


def uniform_truncated_data(batch_size):
    data = np.random.uniform(low=-2, high=2, size=(batch_size, 2))
    data = np.maximum(data, -1)
    data = np.minimum(data, 1)
    return data


def four_corners(_):
    return np.array([
        [-1, 1],
        [-1, -1],
        [1, 1],
        [1, -1],
    ])


def empty_dataset(_):
    return np.zeros((0, 2))


def uniform_gaussian_data(batch_size):
    data = np.random.randn(batch_size, 2)
    data = np.maximum(data, -1)
    data = np.minimum(data, 1)
    return data


def uniform_data(batch_size):
    return np.random.uniform(low=-2, high=2, size=(batch_size, 2))


def affine_gaussian_data(batch_size):
    return (
            np.random.randn(batch_size, 2) * np.array([1, 10]) + np.array(
        [20, 1])
    )


def flower_data(batch_size):
    z_true = np.random.uniform(0, 1, batch_size)
    r = np.power(z_true, 0.5)
    phi = 0.25 * np.pi * z_true
    x1 = r * np.cos(phi)
    x2 = r * np.sin(phi)

    # Sampling form a Gaussian
    x1 = np.random.normal(x1, 0.10 * np.power(z_true, 2), batch_size)
    x2 = np.random.normal(x2, 0.10 * np.power(z_true, 2), batch_size)

    # Bringing data in the right form
    X = np.transpose(np.reshape((x1, x2), (2, batch_size)))
    X = np.asarray(X, dtype='float32')
    return X


def project_samples_square_np(samples):
    samples = np.maximum(samples, -1)
    samples = np.minimum(samples, 1)
    return samples


def project_samples_ell_np(samples):
    samples = project_samples_square_np(samples)
    corners = np.logical_and(samples[:, 0] > 0, samples[:, 1] > 0)

    samples[corners] = np.minimum(samples[corners], 0)
    samples[n:, 1] = np.minimum(samples[n:, 1], 0)
    return samples