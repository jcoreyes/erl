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
from railrl.torch.networks import FlattenMlp, CNN
from railrl.torch import pytorch_util as ptu
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import keras
#
X = np.load('/home/murtaza/vae_data/sawyer_torque_control_ou_imgs_zoomed_out10000.npy')
Y = np.load('/home/murtaza/vae_data/sawyer_torque_control_ou_states_zoomed_out10000.npy')
hidden_sizes = [300, 300, 300]
representation_size =16
num_samples = X.shape[0]
batch_size = 128
# Y = np.concatenate((Y[:, :7], Y[:, 14:]), axis=1)
Y = Y[:, :7] #joint angle regression only
model = FlattenMlp(input_size = representation_size, hidden_sizes=hidden_sizes, output_size=Y.shape[1])

#load vae
# vae = load_vae('/home/murtaza/Documents/rllab/railrl/data/local/05-23-sawyer-torque-vae-with-mse-loss/05-23-sawyer_torque_vae_with_mse_loss_2018_05_23_17_22_02_0000--s-59708/itr_80.pkl')
# vae = load_vae('/home/murtaza/Documents/rllab/railrl/data/local/05-23-sawyer-pos-control-vae-10K/05-23-sawyer-pos_control-vae-10K_2018_05_23_11_39_38_0000--s-823-r16/params.pkl')
# tensor = ptu.np_to_var(X)
# X, log_var = vae.encode(tensor)
# X = ptu.get_numpy(X)
X = np.reshape(X, (num_samples, 84, 84, 3))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# num_epochs = 1000
# loss_fn = nn.MSELoss()
#
# def get_batch(X, Y, batch_size):
#     ind = np.random.randint(0, Y.shape[0], batch_size)
#     X = X[ind, :]
#     Y = Y[ind, :]
#     return ptu.np_to_var(X), ptu.np_to_var(Y)
# losses = []
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for batch_idx in range(100):
#         x_train, y_train = get_batch(X_train, Y_train, batch_size)
#         optimizer.zero_grad()
#         preds = model(x_train)
#         loss = loss_fn(preds, y_train)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss
#     losses.append(ptu.get_numpy(total_loss)[0]/100)
#     print('Average Epoch {} Loss: {} '.format(epoch, ptu.get_numpy(total_loss)[0]/100))
#
# preds = model(ptu.np_to_var(X_test))
# loss = loss_fn(preds, ptu.np_to_var(Y_test))
# print('Test loss: ', ptu.get_numpy(loss)[0])
# plt.plot(losses)

from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.layers import Convolution2D, MaxPooling2D
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


def create_feedforward_network(model, hidden_sizes, input_shape=None):
    count = 0
    for size in hidden_sizes:
        if count == 0 and input_shape is not None:
            model.add(Dense(size, activation='relu', input_shape=input_shape))
            count+=1
        else:
            model.add(Dense(size, activation='relu'))

def create_convolutional_network(model, conv_sizes, fc_sizes, input_shape):
    #structure: conv, pool, conv, pool and so on
    count = 0
    for size in conv_sizes:
        num_filters, kernel_size, pool_size = size
        stride = 1
        if count == 0:
            model.add(Convolution2D(num_filters, strides=(stride, stride), kernel_size=(kernel_size, kernel_size), padding='same', activation='relu', input_shape=input_shape))
            count+=1
        else:
            model.add(Convolution2D(num_filters, strides=(stride, stride), kernel_size=(kernel_size, kernel_size), padding='same', activation='relu'))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
        # model.add(Dropout(.1))

    model.add(Flatten())
    create_feedforward_network(model, fc_sizes)

def create_network(hidden_sizes, num_outputs, do_regression, use_fc=True, conv_sizes=None, input_shape=None, optimizer='adam'):
    model = Sequential()
    if use_fc:
        create_feedforward_network(model, hidden_sizes, input_shape)
    else:
        create_convolutional_network(model, conv_sizes, hidden_sizes, input_shape)
    if do_regression:
        model.add(Dense(num_outputs))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
    else:
        model.add(Dense(num_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

conv_architectures = [
    [[16, 3, 1], [32, 3, 2], [64, 3, 2], [64, 3, 2]],
    # [[16, 5, 3, 2], [32, 3, 1, 1]],
]
fc_architectures = [
    # [256, 128, 64, 32, 16],
    [128, 32],
]

batch_size = 32
num_epochs = 100
histories = []
for conv_architecture in conv_architectures:
    for fc_architecture in fc_architectures:
        model = create_network(fc_architecture, Y_train.shape[1], do_regression=True, conv_sizes = conv_architecture, use_fc=False, input_shape=[84, 84, 3])
        hist = model.fit(X_train, Y_train, batch_size=batch_size, shuffle=True, steps_per_epoch=None, epochs=num_epochs, validation_split=0)
        print(model.evaluate(X_train, Y_train))
        histories.append(hist)
        del model
train_losses = [history.history['loss'] for history in histories]
# val_losses = [history.history['val_loss'] for history in histories]
labels = ['Training']
val_labels = ['Validation']
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_losses[0], label='Training')
# plt.plot(val_losses[0], label='Validation')

# plt.legend()
# plt.show()

X = np.load('/home/murtaza/vae_data/sawyer_torque_control_ou_imgs_zoomed_out_210000.npy')
Y = np.load('/home/murtaza/vae_data/sawyer_torque_control_ou_states_zoomed_out_210000.npy')
X = np.reshape(X, (num_samples, 84, 84, 3))
Y = Y[:,:7]
model.evaluate(X, Y)