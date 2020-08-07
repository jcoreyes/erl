import numpy as np

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import CNN
from rlkit.torch.supervised_learning.supervised_algorithm import SupervisedAlgorithm


def experiment(variant):
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    ptu.set_gpu_mode(True)
    info = dict()
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    net = CNN(**variant['cnn_kwargs'])
    net.cuda()
    num_divisions = variant['num_divisions']
    images = np.zeros((num_divisions * 10000, 21168))
    states = np.zeros((num_divisions*10000, 7))
    for i in range(num_divisions):
        imgs = np.load('/home/murtaza/vae_data/sawyer_torque_control_images100000_' + str(i + 1) + '.npy')
        state = np.load('/home/murtaza/vae_data/sawyer_torque_control_states100000_' + str(i + 1) + '.npy')[:,:7] % (2 * np.pi)
        images[i * 10000:(i + 1) * 10000] = imgs
        states[i * 10000:(i + 1) * 10000] = state
        print(i)
    if variant['normalize']:
        std = np.std(states, axis=0)
        mu = np.mean(states, axis=0)
        states = np.divide((states - mu), std)
        print(mu, std)
    mid = int(num_divisions * 10000 * .9)
    train_images, test_images = images[:mid], images[mid:]
    train_labels, test_labels = states[:mid], states[mid:]


    algo = SupervisedAlgorithm(
        train_images,
        test_images,
        train_labels,
        test_labels,
        net,
        batch_size=variant['batch_size'],
        lr=variant['lr'],
        weight_decay=variant['weight_decay']
    )
    for epoch in range(variant['num_epochs']):
        algo.train_epoch(epoch)
        algo.test_epoch(epoch)

if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'
    use_gpu = True

    variant = dict(
        cnn_kwargs=dict(
        input_width=84,
        input_height=84,
        input_channels=3,
        output_size=7,
        kernel_sizes=[3, 3, 3, 3],
        n_channels=[16, 16, 16, 16],
        strides=[1, 1, 1, 1],
        pool_sizes=[2, 2, 2, 2],
        paddings=[0, 0, 0, 0],
        hidden_sizes=[100],
        use_batch_norm=True,
        ),
        batch_size = 128,
        lr = 3e-4,
        normalize=False,
        num_epochs=500,
        weight_decay=0,
        num_divisions=5,
    )

    search_space = {
        'batch_size':[256],
        'cnn_kwargs.hidden_sizes':[[100], [100, 100]],
        'weight_decay':[0, .001, .01, .1],
        'lr':[1e-2, 1e-3, 1e-4],
        'normalize':[True],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=use_gpu,
                snapshot_mode='gap',
                snapshot_gap=20,
            )

