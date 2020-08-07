import numpy as np

import rlkit.misc.hyperparameter as hyp
from rlkit.envs.vae_wrappers import load_vae
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.supervised_learning.supervised_algorithm import SupervisedAlgorithm


def experiment(variant):
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    info = dict()
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
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
    net = ConcatMlp(input_size=32, hidden_sizes=variant['hidden_sizes'], output_size=states.shape[1])
    vae = variant['vae']
    vae.cuda()
    tensor = ptu.np_to_var(images)
    images, log_var = vae.encode(tensor)
    images = ptu.get_numpy(images)
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
    exp_prefix = 'latent_regression_sweep'
    use_gpu = True

    variant = dict(
        hidden_sizes = [300, 300, 300],
        batch_size = 128,
        lr = 3e-4,
        normalize=True,
        num_epochs=200,
        weight_decay=0,
        num_divisions=1,
        vae = None #load vae here
    )

    search_space = {
        'batch_size':[256],
        'hidden_sizes':[[100], [100, 100], [300, 300, 300]],
        'weight_decay':[.001, .01, .1],
        'lr':[1e-3, 1e-4],
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

