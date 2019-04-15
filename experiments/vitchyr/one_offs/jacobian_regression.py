import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import railrl.torch.pytorch_util as ptu
import torch
from gym.envs.mujoco import PusherEnv, AntEnv, HalfCheetahEnv, HumanoidStandupEnv
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch import nn

from railrl.torch.core import PyTorchModule
from railrl.torch.networks import Mlp, OuterProductFF

plt.ion()

# env = PusherEnv()
# joint_slice = slice(None, 7)
# joint_name = 'tips_arm'

env = HalfCheetahEnv()
joint_slice = slice(None, 7)
joint_name = 'ffoot'

# env = HumanoidStandupEnv()
# joint_slice = slice(None, None)
# joint_name = 'right_foot'

# env = AntEnv()
# joint_slice = slice(2, 13)
# joint_name = 'aux_4'

N_PATHS = 1000
N_PATHS_TEST = N_PATHS
PATH_LENGTH = 10
N_EPOCHS = 50


def generate_data(n_paths, path_length):
    joint_angles = []
    # jacobians = []
    states = []
    actions = []
    next_states = []
    for i in range(n_paths):
        state = env.reset()
        # tip_arm_jac = env.sim.data.get_body_jacp(joint_name)
        for _ in range(path_length):
            joint_angles.append(state[joint_slice])
            # jacobians.append(tip_arm_jac)
            states.append(state)
            action = env.action_space.sample()
            state, *_ = env.step(action)
            actions.append(action)
            next_states.append(state)
            # tip_arm_jac = env.sim.data.get_body_jacp(joint_name)
    # return np.array(joint_angles), np.array(jacobians)
    return (
        np.hstack((np.array(states), np.array(actions))),
        np.array(next_states) - np.array(states)
    )


train_x_np, train_y_np = generate_data(N_PATHS, PATH_LENGTH)
test_x_np, test_y_np = generate_data(N_PATHS_TEST, PATH_LENGTH)
train_x = ptu.np_to_var(train_x_np)
train_y = ptu.np_to_var(train_y_np)
test_x = ptu.np_to_var(test_x_np)
test_y = ptu.np_to_var(test_y_np)

train_dataset = TensorDataset(
    ptu.FloatTensor(train_x_np),
    ptu.FloatTensor(train_y_np),
)
in_dim = train_x_np[0].size
out_dim = train_y_np[0].size
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


def train_network(net, title):
    train_losses = []
    test_losses = []
    times = []

    optimizer = Adam(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for i in range(N_EPOCHS):
        for i_batch, sample_batched in enumerate(dataloader):
            x, y = sample_batched
            x = ptu.Variable(x)
            y = ptu.Variable(y)
            y_hat = net(x)

            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_hat = net(test_x)
        test_loss = float(criterion(y_hat, test_y))
        test_losses.append(test_loss)

        y_hat = net(train_x)
        train_loss = float(criterion(y_hat, train_y))
        train_losses.append(train_loss)

        times.append(i)
        plt.gcf().clear()
        plt.plot(times, train_losses, '--')
        plt.plot(times, test_losses, '-')
        plt.title(title)
        plt.draw()
        plt.pause(0.05)
    print(title)
    print("\tfinal train loss: {}".format(train_loss))
    print("\tfinal test loss: {}".format(test_loss))


class JacobianNet(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.fcs = []
        self.gates = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            fc = nn.Linear(in_size, next_size)
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("gate{}".format(i), fc)
            self.gates.append(fc)

            in_size = next_size

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        z = input
        for fc, gate in zip(self.fcs, self.gates):
            z = torch.sin(gate(z))
            h = fc(h) * z
        return self.last_fc(h)


def num_params(net):
    return nn.utils.parameters_to_vector(net.parameters()).shape[0]

mean_y = np.mean(test_y_np, axis=0)
print("Mean y", mean_y)
print("Constant error", np.mean((test_y_np - mean_y)**2))

plt.figure()
mlp = Mlp(hidden_sizes=[100, 100], output_size=out_dim, input_size=in_dim)

mlp_n_params = num_params(mlp)

h_size = 100
jac_net = JacobianNet(hidden_sizes=[h_size, h_size], output_size=out_dim,
                      input_size=in_dim)
# keep # paramers ~ same
while num_params(jac_net) > mlp_n_params:
    h_size -= 5
    jac_net = JacobianNet(hidden_sizes=[h_size, h_size], output_size=out_dim,
                          input_size=in_dim)
print("jac_net h_size:", h_size)

linear_net = Mlp(hidden_sizes=[], output_size=out_dim, input_size=in_dim)


sin_mlp = Mlp(hidden_sizes=[100, 100], output_size=out_dim, input_size=in_dim,
          hidden_activation=torch.sin)
tanh_mlp = Mlp(hidden_sizes=[100, 100], output_size=out_dim, input_size=in_dim,
          hidden_activation=torch.tanh)
plt.figure()
train_network(mlp, "mlp [100, 100]")
plt.figure()
train_network(jac_net, "jac [{h_size}, {h_size}]".format(h_size=h_size))
# plt.figure()
# train_network(linear_net, "linear")
# plt.figure()
# train_network(sin_mlp, "sin [100, 100]")
# plt.figure()
# train_network(tanh_mlp, "tanh [100, 100]")













import ipdb; ipdb.set_trace()
