'''
retrain vae, store images, qpos
encode images into X
Y = qpos
train a neural net to predict
'''
#assume we have X and Y for now
import torch
from torch import optim, nn
import numpy as np

from railrl.envs.vae_wrappers import load_vae
from railrl.torch.networks import FlattenMlp, CNN
from railrl.torch import pytorch_util as ptu
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import keras
#
X = np.load('/home/murtaza/vae_data/sawyer_torque_control_ou_imgs_zoomed_out10000.npy')
Y = np.load('/home/murtaza/vae_data/sawyer_torque_control_ou_states_zoomed_out10000.npy')
hidden_sizes = [300, 300, 300]
representation_size =32
num_samples = X.shape[0]
batch_size = 512
# Y = np.concatenate((Y[:, :7], Y[:, 14:]), axis=1)
Y = Y[:, :7] % (2*np.pi) #joint angle regression only

#load vae
# vae = load_vae('/home/murtaza/Documents/rllab/railrl/data/local/05-23-sawyer-torque-vae-with-mse-loss/05-23-sawyer_torque_vae_with_mse_loss_2018_05_23_17_22_02_0000--s-59708/itr_80.pkl')
vae = load_vae('/home/murtaza/Documents/rllab/railrl/experiments/murtaza/vae/torque_control/debug_torque_params.pkl')
tensor = ptu.np_to_var(X)
X, log_var = vae.encode(tensor)
X = ptu.get_numpy(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1)
num_epochs = 1000
loss_fn = nn.MSELoss()
model = FlattenMlp(input_size = representation_size, hidden_sizes=hidden_sizes, output_size=Y.shape[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def get_batch(X, Y, batch_size):
    ind = np.random.randint(0, Y.shape[0], batch_size)
    X = X[ind, :]
    Y = Y[ind, :]
    return ptu.np_to_var(X), ptu.np_to_var(Y)
losses = []
per_dim_errors = np.zeros((num_epochs, Y.shape[1]))
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    per_dim_total_error = 0
    for batch_idx in range(100):
        x_train, y_train = get_batch(X_train, Y_train, batch_size)
        optimizer.zero_grad()
        preds = model(x_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss
        per_dim_total_error += torch.abs(y_train - preds)
    losses.append(ptu.get_numpy(total_loss)[0]/100)
    per_dim_errors[epoch] = ptu.get_numpy(per_dim_total_error)[0]/100
    print('Per Dim errors: ',ptu.get_numpy(per_dim_total_error)[0]/100 )
    print('Average Epoch {} Loss: {} '.format(epoch, ptu.get_numpy(total_loss)[0]/100))
    preds = model(ptu.np_to_var(X_test))
    loss = loss_fn(preds, ptu.np_to_var(Y_test))
    print('Test loss: ', ptu.get_numpy(loss)[0])
plt.plot(losses)
plt.show()

