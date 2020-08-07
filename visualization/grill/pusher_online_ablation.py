from collections import OrderedDict

import matplotlib
from visualization.grill.config import (
    output_dir,
    vitchyr_base_dir,
    format_func,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from rlkit.visualization import plot_util as plot
from rlkit.misc import data_processing as dp

configure_matplotlib(matplotlib)

pusher_dir = vitchyr_base_dir + 'papers/nips2018/pusher-online-vae'

online_pusher = dp.get_trials(
    pusher_dir,
    criteria={
        'rdim': 250,
        'algo_kwargs.should_train_vae.$function': 'railrl.torch.vae.vae_schedules.every_three',
    }
)
offline_pusher = dp.get_trials(
    pusher_dir,
    criteria={
        'rdim': 250,
        'algo_kwargs.should_train_vae.$function': 'railrl.torch.vae.vae_schedules.never_train',
    }
)
plt.figure(figsize=(6, 5))
plot.plot_trials(
    OrderedDict([
        ("Online", online_pusher),
        ("Offline", offline_pusher),
    ]),
    y_keys="Final  sum_distance Mean",
    x_key="Number of env steps total",
    process_time_series=plot.padded_ma_filter(100),
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.title("Visual Pusher, Online Ablation")
lgnd = plt.legend(["Online", "Offline"], bbox_to_anchor=(0.49, -0.2), loc="upper center", ncol=4, handlelength=1)
plt.tight_layout()
plt.savefig(output_dir + "pusher_online_ablation.pdf")
print("File saved to", output_dir + "pusher_online_ablation.pdf")
