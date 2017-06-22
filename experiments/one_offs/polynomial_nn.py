import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms
from torch.autograd import Variable
import railrl.misc.visualization_util as vu

from railrl.pythonplusplus import line_logger


class BatchMatrixMultiply(nn.Module):
    def __init__(self, state_size, size):
        super().__init__()
        self.state_size = size
        self.size = size
        self.L = nn.Linear(state_size, size**2)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)
        self.tril_mask = Variable(torch.tril(torch.ones(size, size), k=-1).unsqueeze(0))
        self.diag_mask = Variable(torch.diag(torch.diag(torch.ones(size, size))).unsqueeze(0))

    def forward(self, x, term):
        L = self.L(x).view(-1, self.size, self.size)
        L = L * (
            self.tril_mask.expand_as(L)
            + torch.exp(L) * self.diag_mask.expand_as(L)
        )
        P = torch.bmm(L, L.transpose(2, 1))
        term = term.unsqueeze(2)
        # return torch.bmm(torch.bmm(term.transpose(2, 1), P), term).squeeze(2)
        return torch.bmm(term.transpose(2, 1), P).squeeze(1)


class Polynomial(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.vf_fc1_size = 100
        self.vf_fc2_size = 200
        self.vf = nn.Sequential(
            nn.Linear(self.obs_dim, self.vf_fc1_size),
            nn.Tanh(),
            nn.Linear(self.vf_fc1_size, self.vf_fc2_size),
            nn.Tanh(),
            nn.Linear(self.vf_fc2_size, 1),
        )

        self.bmm1 = BatchMatrixMultiply(self.obs_dim, self.action_dim)
        self.bmm2 = BatchMatrixMultiply(self.obs_dim, self.action_dim)
        self.bmm3 = BatchMatrixMultiply(self.obs_dim, self.action_dim)

    def forward(self, state, action):
        V = self.vf(state)

        h = self.bmm1(state, action)
        h = self.bmm2(state, h + action)
        h = self.bmm2(state, h + action)

        A = torch.bmm(h.unsqueeze(1), action.unsqueeze(2)).squeeze(2)
        A = F.relu(A)

        return A, V


class FFModel(nn.Module):
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


def q_function(state, action):
    return state**2 + state * action


class FakeDataset(data.Dataset):
    def __init__(self, obs_dim, action_dim, size):
        self.size = size
        self.state = torch.rand(size, obs_dim)
        self.action = torch.rand(size, action_dim)
        self.q_value = torch.sum(
            q_function(self.state, self.action), dim=1,
        )

    def __getitem__(self, index):
        return self.state[index], self.action[index], self.q_value[index]

    def __len__(self):
        return self.size


def main():
    obs_dim = 2
    action_dim = 2
    batch_size = 32
    model = Polynomial(obs_dim, action_dim)
    # model = FFModel(obs_dim, action_dim)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    loss_fnct = nn.MSELoss()

    num_batches_per_print = 100
    train_size = 100000
    test_size = 10000

    train_loader = data.DataLoader(
        FakeDataset(obs_dim, action_dim, train_size),
        batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(
        FakeDataset(obs_dim, action_dim, test_size),
        batch_size=batch_size, shuffle=True)

    def eval_model(state, action):
        state = Variable(state, requires_grad=False)
        action = Variable(action, requires_grad=False)
        a, v = model(state, action)
        return a + v


    def train(epoch):
        for batch_idx, (state, action, q_target) in enumerate(train_loader):
            q_estim = eval_model(state, action)
            q_target = Variable(q_target, requires_grad=False)

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
            state = Variable(state, requires_grad=False)
            action = Variable(action, requires_grad=False)
            q_target = Variable(q_target, requires_grad=False)

            a, v = model(state, action)
            q_estim = a + v
            loss = loss_fnct(q_estim, q_target)
            test_losses.append(loss.data[0])

        line_logger.newline()
        print('Test Epoch: {0}. Loss: {1}'.format(epoch, np.mean(test_losses)))

    def visualize_model():
        state_bounds = (-10, 10)
        action_bounds = (-10, 10)
        resolution = 10
        true_heatmap = vu.make_heat_map(
            q_function,
            x_bounds=state_bounds,
            y_bounds=action_bounds,
            resolution=resolution,
        )
        estimated_heatmap = vu.make_heat_map(

        )

    for epoch in range(1, 10):
        model.train()
        train(epoch)
        model.eval()
        test(epoch)

    # visualize_model()

if __name__ == '__main__':
    main()
