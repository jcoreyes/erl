import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import time

x_dim = 5
y_dim = 5
num_iters = 100
batch_size = 32
num_steps = 100

def get_batch():
    X = np.random.rand(batch_size, num_steps, x_dim)
    y = np.random.rand(batch_size, num_steps, y_dim)
    return X, y

def get_tf_time():
    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.float32, shape=[None, num_steps, x_dim])
    y_ph = tf.placeholder(tf.float32, shape=[None, num_steps, y_dim])
    inputs_unstacked = tf.unstack(x_ph, axis=1)
    cell = LSTMCell(y_dim)
    rnn_outputs, rnn_final_state = tf.contrib.rnn.static_rnn(
        cell,
        inputs_unstacked,
        dtype=tf.float32,
    )
    all_rnn_outputs = tf.stack(rnn_outputs, axis=1)
    loss = tf.reduce_mean((y_ph - all_rnn_outputs)**2)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for _ in range(num_iters):
        X, y = get_batch()
        sess.run(train_op, {x_ph: X, y_ph: y})
    return time.time() - start_time

def get_pt_time(cuda=True):
    lstm = nn.LSTM(x_dim, y_dim, 1, batch_first=True)
    if cuda:
        lstm = lstm.cuda()
    criterion = nn.MSELoss()
    opt = optim.Adam(lstm.parameters(), 1e-3)
    start_time = time.time()
    for _ in range(num_iters):
        X, y = get_batch()
        if cuda:
            input = Variable(torch.from_numpy(X)).float().cuda()
            target = Variable(torch.from_numpy(y)).float().cuda()
            h0 = Variable(torch.randn(1, batch_size, y_dim)).float().cuda()
            c0 = Variable(torch.randn(1, batch_size, y_dim)).float().cuda()
        else:
            input = Variable(torch.from_numpy(X)).float()
            target = Variable(torch.from_numpy(y)).float()
            h0 = Variable(torch.randn(1, batch_size, y_dim)).float()
            c0 = Variable(torch.randn(1, batch_size, y_dim)).float()
        outputs, _ = lstm(input, (h0, c0))
        loss = criterion(outputs, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return time.time() - start_time


def get_pt_unrolled_time(cuda=True):
    lstm_cell = nn.LSTMCell(x_dim, y_dim)
    if cuda:
        lstm_cell = lstm_cell.cuda()
    criterion = nn.MSELoss()
    opt = optim.Adam(lstm_cell.parameters(), 1e-3)
    start_time = time.time()
    for _ in range(num_iters):
        X, y = get_batch()
        if cuda:
            input = Variable(torch.from_numpy(X)).float().cuda()
            target = Variable(torch.from_numpy(y)).float().cuda()
            hx = Variable(torch.randn(batch_size, y_dim)).float().cuda()
            cx = Variable(torch.randn(batch_size, y_dim)).float().cuda()
        else:
            input = Variable(torch.from_numpy(X)).float()
            target = Variable(torch.from_numpy(y)).float()
            hx = Variable(torch.randn(batch_size, y_dim)).float()
            cx = Variable(torch.randn(batch_size, y_dim)).float()
        output = []
        for i in range(num_steps):
            hx, cx = lstm_cell(input[:, i, :], (hx, cx))
            output.append(hx)
        outputs = torch.cat(output, dim=1)
        loss = criterion(outputs, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return time.time() - start_time

def main():
    print("GPU")
    print(get_tf_time())
    print(get_pt_time())
    print(get_pt_unrolled_time())

    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    print("No GPU")
    print(get_tf_time())
    print(get_pt_time(cuda=False))
    print(get_pt_unrolled_time(cuda=False))

if __name__ == '__main__':
    main()
