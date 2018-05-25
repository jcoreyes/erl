import numpy as np
import matplotlib.pyplot as plt


def cg(A, b, x0, tolerance):
    iterates = [x0]
    r = b - A @ x0
    p = r
    x = x0
    error = np.linalg.norm(r)
    errors = [error]
    iters = 0
    while error >= tolerance and iters < 100:
        iters += 1
        alpha = (r.T @ r) / (p.T @ A @ p)
        x_new = x + alpha * p
        r_new = r - alpha * (A @ p)
        beta = (r_new.T @ r_new) / (r.T @ r)
        p_new = r_new + beta * p

        x = x_new
        r = r_new
        p = p_new

        iterates.append(x)
        error = np.linalg.norm(r)
        errors.append(error)
    return iterates, errors


def gd(A, b, x0, tolerance, alpha=1e-1):
    x = x0
    iters = 0
    error = np.linalg.norm((A@x) - b)
    iterates = [x]
    errors = [error]
    alpha_0 = alpha
    while error >= tolerance and iters < 100:
        iters += 1

        grad = A @ x - b
        x -= alpha * grad
        alpha = alpha_0 / (iters + 1)

        error = np.linalg.norm((A@x) - b)
        errors.append(error)
        iterates.append(x)
    return iterates, errors


def main():
    M = np.random.rand(10, 10)
    A = M.T @ M
    b = np.random.rand(10, 1)
    x0 = np.zeros_like(b)
    tolerance = 1e-10

    iterates, errors = cg(A, b, x0, tolerance)
    iterates_gd, errors_gd = gd(A, b, x0, tolerance)

    iters = np.arange(len(errors))
    iters_gd = np.arange(len(errors_gd))
    plt.plot(iters, errors, label='Conjugate Gradient')
    plt.plot(iters_gd, errors_gd, label='Gradient Descent')
    plt.xlabel("Iteration")
    plt.ylabel("L2 Error")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
