from railrl.torch.networks import CNN
import torch.nn as nn
import numpy as np
import railrl.torch.pytorch_util as ptu

def random_batch(images, labels, batch_size=64):
    idxs = np.random.choice(len(images), batch_size)
    return images[idxs], labels[idxs]

ptu.set_gpu_mode(True)
imsize=84
args = dict(
    input_width=imsize,
    input_height=imsize,
    input_channels=3,
    output_size=7,
    kernel_sizes=[5, 5, 5],
    n_channels=[32, 32, 64],
    strides=[3, 3, 3],
    pool_sizes=[1, 1, 1],
    paddings=[0, 0, 0],
    hidden_sizes=[400, 300, 300],
    use_batch_norm=True,
)


net = CNN(**args)
net.cuda()
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=0)
weight_decays = [0.05, 0.01, .001, 0]

running_loss = 0
images = np.load('/home/murtaza/vae_data/sawyer_torque_control_ou_imgs_zoomed_out10000.npy')
joint_angles = np.load('/home/murtaza/vae_data/sawyer_torque_control_ou_states_zoomed_out10000.npy')
joint_angles = joint_angles[:, :7] % (2*np.pi)
std = np.std(joint_angles, axis=0)
mu = np.mean(joint_angles, axis=0)
joint_angles = np.divide((joint_angles - mu), std)
train_images, test_images = images[:9000], images[9000:]
train_labels, test_labels = joint_angles[:9000], joint_angles[9000:]

print('std=', std)
val_losses = []
train_losses = []
num_epochs = 300
num_batches = 128
batch_size = 64
for epoch in range(num_epochs):  # loop over the dataset multiple times
    for batch in range(num_batches):
        net.train()
        inputs_np, labels_np = random_batch(train_images, train_labels, batch_size=batch_size)
        inputs, labels = ptu.Variable(ptu.from_numpy(inputs_np)), ptu.Variable(ptu.from_numpy(labels_np))
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.data[0]
    if epoch % 10 == 0:
        net.eval()
        inputs_np, labels_np = random_batch(test_images, test_labels, batch_size=256)
        inputs, labels = ptu.Variable(ptu.from_numpy(inputs_np)), ptu.Variable(ptu.from_numpy(labels_np))
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        val_loss = loss.data[0]
        training_loss = running_loss / num_batches
        val_losses.append(val_loss)
        train_losses.append(training_loss)
        print('Iteration ', epoch, 'validation loss: ', val_loss, 'training loss: ', training_loss)
    running_loss = 0

def compute_output_size(num_samples, h_in, w_in, kernel_size, padding, stride, c_out):
    h_out = (h_in + 2 * padding - (kernel_size-1) - 1)/stride + 1
    w_out = (w_in + 2 * padding - (kernel_size-1) - 1)/stride + 1
    return num_samples, c_out, h_out, w_out