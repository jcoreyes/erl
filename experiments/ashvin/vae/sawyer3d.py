from railrl.torch.vae.conv_vae import ConvVAE, ConvVAETrainer
from railrl.torch.vae.sawyer3D_data import get_data
from railrl.launchers.launcher_util import run_experiment
from railrl.launchers.arglauncher import run_variants
import railrl.torch.pytorch_util as ptu

def experiment(variant):
    if variant["use_gpu"]:
        ptu.set_gpu_mode(True)
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data = get_data()
    m = ConvVAE(representation_size, input_channels=3)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta)

    for epoch in range(1001):
        t.train_epoch(epoch)
        t.test_epoch(epoch)
        t.dump_samples(epoch)

if __name__ == "__main__":
    variants = []

    for representation_size in [2, 4, 8, 16, 32, 64]:
        for beta in [5.0]:
            variant = dict(
                beta=beta,
                representation_size=representation_size,
                use_gpu=True,
                mode='here_no_doodad'
            )
            variants.append(variant)
    for variant in variants:
        n_seeds = 1
        exp_prefix = 'sawyer_vae_train'
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                snapshot_mode='gap',
                snapshot_gap=20,
                exp_prefix=exp_prefix,
                variant=variant
            )
    run_variants(experiment, variants, run_id=1)