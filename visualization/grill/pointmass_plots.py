import matplotlib
from visualization.grill.config import (
    output_dir,
    format_func,
    our_method_name,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from railrl.misc import plot_util as plot
from railrl.misc import data_processing as dp

configure_matplotlib(matplotlib)

dirs = [
    '/home/khazatsky/rail/data/rail-khazatsky/sasha/cond-rig/pointmass/baseline',
]
VAE = plot.load_exps(dirs, suppress_output=True)
plot.tag_exps(VAE, "name", "VAE")

dirs = [
    '/home/khazatsky/rail/data/rail-khazatsky/sasha/cond-rig/pointmass/standard',
]
f = plot.exclude_by_flat_params({'grill_variant.vae_path': '/home/khazatsky/rail/data/rail-khazatsky/sasha/PCVAE/standard/run1002/id0/vae.pkl',})
CVAE = plot.load_exps(dirs, f, suppress_output=True)
plot.tag_exps(CVAE, "name", "CVAE")

dirs = [
    '/home/khazatsky/rail/data/rail-khazatsky/sasha/cond-rig/pointmass/dynamics/run12',
       ]
CDVAE = plot.load_exps(dirs, suppress_output=True)
plot.tag_exps(CDVAE, "name", "Dynamics CVAE")


dirs = [
    '/home/khazatsky/rail/data/rail-khazatsky/sasha/cond-rig/pointmass/adversarial/run10/',
       ]
f = plot.filter_by_flat_params({'grill_variant.vae_path': '/home/khazatsky/rail/data/rail-khazatsky/sasha/PCVAE/ACE/run1000/id1/vae.pkl',})
ACE = plot.load_exps(dirs, f, suppress_output=True)
plot.tag_exps(ACE, "name", "Adversarially Conditioned VAE")

dirs = [
    '/home/khazatsky/rail/data/rail-khazatsky/sasha/cond-rig/pointmass/CADVAE/run10/',
       ]

CADVAE = plot.load_exps(dirs, suppress_output=True)
plot.tag_exps(CADVAE, "name", "Adversarially Conditioned Dynamics VAE")

dirs = [
    '/home/khazatsky/rail/data/rail-khazatsky/sasha/cond-rig/pointmass/standard/',
]
f = plot.filter_by_flat_params({'grill_variant.vae_path': '/home/khazatsky/rail/data/rail-khazatsky/sasha/PCVAE/standard/run1001/id0/vae.pkl',})
DCVAE = plot.load_exps(dirs, f, suppress_output=True)
plot.tag_exps(DCVAE, "name", "Delta CVAE")



plot.comparison(
    VAE + CVAE + CDVAE + ACE + CADVAE + DCVAE,
    ["evaluation/Final distance_to_target Mean",],
    vary=["name"],
    smooth=plot.padded_ma_filter(10),
    ylim=(0.1, 6),
    xlim=(0, 800),
    figsize=(20, 10),
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Epochs")
plt.ylabel("Distance To Goal")
plt.title("Visual Pointmass Distance")
plt.legend(["Adversarially Conditioned Dynamics VAE", "Adversarially Conditioned VAE", "CVAE", "Delta CVAE", "Dynamics CVAE", "VAE"],loc="upper right") # [our_method_name, "DSAE", "HER", "Oracle", "L&R", ])

plt.tight_layout()
plt.savefig(output_dir + "pointmass_distance.pdf")
print("File saved to", output_dir + "pointmass_distance.pdf")
