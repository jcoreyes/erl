'''
retrain vae, store images, qpos
encode images into X
Y = qpos
train a neural net to predict
'''
#assume we have X and Y for now
from torch import optim, nn
import numpy as np

from railrl.envs.vae_wrappers import load_vae
from railrl.torch.networks import FlattenMlp
from railrl.torch import pytorch_util as ptu
from sklearn.model_selection import train_test_split

X = np.load('/tmp/sawyer_torque_control_imgs10000.npy')
Y = np.load('/tmp/sawyer_torque_control_states10000.npy')
hidden_sizes = [300, 300]
representation_size =16
num_samples = 10000
batch_size = 1024
model = FlattenMlp(input_size = representation_size, hidden_sizes=hidden_sizes, output_size=Y.shape[1])
#load vae
vae = load_vae('/home/murtaza/Documents/rllab/railrl/data/local/05-22-sawyer-torque-reacher-vae-10K/05-22-sawyer_torque_reacher_vae_10K_2018_05_22_16_54_56_0000--s-15151/params.pkl')
tensor = ptu.np_to_var(X)
X, log_var = vae.encode(tensor)
X = ptu.get_numpy(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 1000
loss_fn = nn.MSELoss()

def get_batch(X, Y, batch_size):
    ind = np.random.randint(0, Y.shape[0], batch_size)
    X = X[ind, :]
    Y = Y[ind, :]
    return ptu.np_to_var(X), ptu.np_to_var(Y)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx in range(100):
        x_train, y_train = get_batch(X_train, Y_train, batch_size)
        optimizer.zero_grad()
        preds = model(x_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss
    print('Average Epoch {} Loss: {} '.format(epoch, ptu.get_numpy(total_loss)[0]/100))

preds = model(ptu.np_to_var(X_test))
loss = loss_fn(preds, ptu.np_to_var(Y_test))
print('Test loss: ', ptu.get_numpy(loss)[0])