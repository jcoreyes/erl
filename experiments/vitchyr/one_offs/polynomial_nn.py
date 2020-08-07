import numpy as np
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import rlkit.misc.visualization_util as vu
import matplotlib.pyplot as plt
import seaborn as sns

from rlkit.misc.html_report import HTMLReport
from rlkit.pythonplusplus import line_logger
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.modules import SelfOuterProductLinear
from rlkit.torch.pytorch_util import double_moments


class BatchSquare(nn.Module):
    """
    Compute x^T P(s) x
    """
    def __init__(self, matrix_input_size, vector_size):
        super().__init__()
        self.vector_size = vector_size

        self.L = nn.Linear(matrix_input_size, vector_size ** 2)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)
        self.tril_mask = ptu.Variable(
            torch.tril(torch.ones(vector_size, vector_size), k=-1).unsqueeze(0)
        )
        self.diag_mask = ptu.Variable(
            torch.diag(torch.diag(torch.ones(vector_size, vector_size))).unsqueeze(0)
        )

    def forward(self, state, vector):
        L = self.L(state).view(-1, self.vector_size, self.vector_size)
        L = L * (
            self.tril_mask.expand_as(L)
            + torch.exp(L) * self.diag_mask.expand_as(L)
        )
        P = torch.bmm(L, L.transpose(2, 1))
        vector = vector.unsqueeze(2)
        return torch.bmm(
            torch.bmm(vector.transpose(2, 1), P), vector
        ).squeeze(2)


class NAF(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.embed_fc1_size = 100
        self.embed_size = 200

        self.embed = nn.Sequential(
            nn.Linear(self.obs_dim, self.embed_fc1_size),
            nn.Tanh(),
            nn.Linear(self.embed_fc1_size, self.embed_size),
            nn.Tanh(),
        )

        self.mu = nn.Linear(self.embed_size, action_dim)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.vf = nn.Sequential(
            nn.Linear(self.embed_size, 1),
        )

        self.bs = BatchSquare(self.embed_size, self.action_dim)
        # self.bs = BatchSquare(self.obs_dim, self.action_dim)

    def forward(self, state, action):
        embedded = self.embed(state)

        V = self.vf(embedded)
        mu = torch.tanh(self.mu(embedded))
        A = self.bs(embedded, action - mu)
        # A = self.bs(state, action - mu)

        return A, V


class SeparateDuelingFF(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.vf_fc1_size = 100
        self.vf_fc2_size = 200
        self.af_fc1_size = 100
        self.af_fc2_size = 200

        self.vf = nn.Sequential(
            nn.Linear(self.obs_dim, self.vf_fc1_size),
            nn.Tanh(),
            nn.Linear(self.vf_fc1_size, self.vf_fc2_size),
            nn.Tanh(),
            nn.Linear(self.vf_fc2_size, 1),
        )

        self.af = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.af_fc1_size),
            nn.Tanh(),
            nn.Linear(self.af_fc1_size, self.af_fc2_size),
            nn.Tanh(),
            nn.Linear(self.af_fc2_size, 1),
        )

    def forward(self, state, action):
        V = self.vf(state)
        A = self.af(torch.cat((state, action), dim=1))
        return A, V


class ConcatFF(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.fc1_size = 100
        self.fc2_size = 100

        self.af = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.fc1_size),
            nn.Tanh(),
            nn.Linear(self.fc1_size, self.fc2_size),
            nn.Tanh(),
            nn.Linear(self.fc2_size, 1),
        )

    def forward(self, state, action):
        A = self.af(torch.cat((state, action), dim=1))
        return A, 0


class OuterProductFF(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.fc1_size = 100
        self.fc2_size = 100

        self.af = nn.Sequential(
            SelfOuterProductLinear(self.obs_dim + self.action_dim,
                      self.fc1_size),
            nn.Tanh(),
            SelfOuterProductLinear(self.fc1_size, self.fc2_size),
            # nn.Linear(self.fc1_size, self.fc2_size),
            nn.Tanh(),
            SelfOuterProductLinear(self.fc2_size, 1),
            # nn.Linear(self.fc2_size, 1),
        )

    def forward(self, state, action):
        flat = torch.cat((state, action), dim=1)
        # h = outer_product(flat, flat)
        A = self.af(flat)
        return A, 0


def q_function_torch(state, action):
    # return state**2 + action**2
    return action**2


def q_function(state, action):
    # return state**2# + action**2
    return action**2


def uniform(size, bounds):
    values = torch.rand(*size)
    min_value, max_value = bounds
    delta = max_value - min_value
    values *= delta
    values += min_value
    return values


class FakeDataset(data.Dataset):
    def __init__(self, obs_dim, action_dim, size, state_bounds, action_bounds):
        self.size = size
        self.state = uniform((size, obs_dim), state_bounds)
        self.action = uniform((size, action_dim), action_bounds)
        self.q_value = torch.sum(
            q_function_torch(self.state, self.action), dim=1,
        )

    def __getitem__(self, index):
        return self.state[index], self.action[index], self.q_value[index]

    def __len__(self):
        return self.size


def main():
    ptu.set_gpu_mode(True)

    obs_dim = 1
    action_dim = 1
    batch_size = 100

    model = NAF(obs_dim, action_dim)
    # model = SeparateDuelingFF(obs_dim, action_dim)
    # model = ConcatFF(obs_dim, action_dim)
    # model = OuterProductFF(obs_dim, action_dim)
    version = model.__class__.__name__
    version = "NAF-P-depends-on-embedded"

    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.5)
    loss_fnct = nn.MSELoss()

    num_batches_per_print = 100
    train_size = 100000
    test_size = 10000

    state_bounds = (-10, 10)
    action_bounds = (-10, 10)
    resolution = 20

    base_dir = Path(
        "/home/vitchyr/git/rllab-rail/railrl/data/one-offs/polynomial-nn"
    )
    base_dir = base_dir / version
    if not base_dir.exists():
        base_dir.mkdir()
    report_path = str(base_dir / "report.html")
    report = HTMLReport(report_path, images_per_row=2)
    print("Saving report to: {}".format(report_path))

    train_loader = data.DataLoader(
        FakeDataset(obs_dim, action_dim, train_size, state_bounds, action_bounds),
        batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(
        FakeDataset(obs_dim, action_dim, test_size, state_bounds, action_bounds),
        batch_size=batch_size, shuffle=True)

    model.to(ptu.device)

    def eval_model(state, action):
        state = ptu.Variable(state, requires_grad=False)
        action = ptu.Variable(action, requires_grad=False)
        a, v = model(state, action)
        return a + v

    def train(epoch):
        for batch_idx, (state, action, q_target) in enumerate(train_loader):
            q_estim = eval_model(state, action)
            q_target = ptu.Variable(q_target, requires_grad=False)

            loss = loss_fnct(q_estim, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % num_batches_per_print == 0:
                line_logger.print_over(
                    'Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, batch_size*batch_idx, train_size, loss.data[0]
                    )
                )

    def test(epoch):
        test_losses = []
        for state, action, q_target in test_loader:
            q_estim = eval_model(state, action)
            q_target = ptu.Variable(q_target, requires_grad=False)
            loss = loss_fnct(q_estim, q_target)
            test_losses.append(loss.data[0])

        line_logger.newline()
        print('Test Epoch: {0}. Loss: {1}'.format(epoch, np.mean(test_losses)))

        report.add_header("Epoch = {}".format(epoch))

        fig = visualize_model(q_function, "True Q Function")
        img = vu.save_image(fig)
        report.add_image(img, txt='True Q Function')

        fig = visualize_model(eval_model_np, "Estimated Q Function")
        img = vu.save_image(fig)
        report.add_image(img, txt='Estimated Q Function')

        report.new_row()

    def eval_model_np(state, action):
        state = ptu.Variable(ptu.FloatTensor([[state]]), requires_grad=False)
        action = ptu.Variable(ptu.FloatTensor([[action]]), requires_grad=False)
        a, v = model(state, action)
        q = a + v
        return ptu.get_numpy(q)[0]

    def visualize_model(eval, title):
        fig = plt.figure()
        ax = plt.gca()
        heatmap = vu.make_heat_map(
            eval,
            x_bounds=state_bounds,
            y_bounds=action_bounds,
            resolution=resolution,
        )

        vu.plot_heatmap(heatmap, fig, ax)
        ax.set_xlabel("State")
        ax.set_ylabel("Action")
        ax.set_title(title)

        return fig

    for epoch in range(0, 10):
        model.train()
        train(epoch)
        model.eval()
        test(epoch)

    print("Report saved to: {}".format(report_path))

if __name__ == '__main__':
    main()
