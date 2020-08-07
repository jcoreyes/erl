"""
Visualize Q-values and gradients of a good Ant policy.
"""
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import rlkit.torch.pytorch_util as ptu

PATH = '/home/vitchyr/git/railrl/data/doodads3/01-26-ddpg-sweep-harder-tasks/01-26-ddpg-sweep-harder-tasks-id10-s68123/params.pkl'
BAD_POLICY_PATH = '/home/vitchyr/git/railrl/data/doodads3/01-26-ddpg-sweep-harder-tasks/01-26-ddpg-sweep-harder-tasks-id3-s32528/params.pkl'
UNSTABLE_POLCIY_PATH = '/home/vitchyr/git/railrl/data/doodads3/01-26-ddpg-sweep-harder-tasks/01-26-ddpg-sweep-harder-tasks-id9-s20629/params.pkl'


def visualize_qf_slice(qf, env):
    ob = env.reset()
    sampled_action = env.action_space.sample()
    low = env.action_space.low
    high = env.action_space.high
    N = 100

    a0_values, da = np.linspace(low[0], high[0], N, retstep=True)
    num_dim = low.size
    for i in range(num_dim):
        q_values = []
        q_gradients = []
        action = sampled_action.copy()
        for a0_value in a0_values:
            action[i] = a0_value
            ob_pt = ptu.np_to_var(ob[None])
            action_pt = ptu.np_to_var(action[None], requires_grad=True)
            q_val = qf(ob_pt, action_pt)
            q_val.sum().backward()

            q_values.append(ptu.get_numpy(q_val)[0, 0])
            q_gradients.append(ptu.get_numpy(action_pt.grad)[0, i])

        q_values = np.array(q_values)
        q_gradients = np.array(q_gradients)
        empirical_gradients = np.gradient(q_values, da)
        plt.subplot(num_dim, 2, i*2 + 1)
        plt.plot(a0_values, q_values, label='values')
        plt.xlabel("action slice")
        plt.ylabel("q-value")
        plt.title("dimension {}".format(i))
        plt.subplot(num_dim, 2, i*2 + 2)
        plt.plot(a0_values, empirical_gradients, label='empirical gradients')
        plt.plot(a0_values, q_gradients, label='actual gradients')
        plt.xlabel("action slice")
        plt.ylabel("dq/da")
        plt.title("dimension {}".format(i))
        plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        default=UNSTABLE_POLCIY_PATH,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=30, help='Horizon for eval')
    args = parser.parse_args()

    data = joblib.load(args.file)
    qf = data['qf']
    env = data['env']
    visualize_qf_slice(qf, env)
