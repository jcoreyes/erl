from collections import OrderedDict

import matplotlib
from visualization.grill.config import (
    output_dir,
    vitchyr_base_dir,
    format_func,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from railrl.visualization import plot_util as plot
from railrl.misc import data_processing as dp

configure_matplotlib(matplotlib)

reacher_dir = vitchyr_base_dir + 'papers/nips2018/reacher_online_vae'
online_reacher = dp.get_trials(
    reacher_dir,
    criteria={
        'algo_kwargs.should_train_vae.$function': 'railrl.torch.vae.vae_schedules.always_train',
    }
)
offline_reacher = dp.get_trials(
    reacher_dir,
    criteria={
        'algo_kwargs.should_train_vae.$function': 'railrl.torch.vae.vae_schedules.never_train',
    }
)

plt.figure(figsize=(6, 5))
plot.plot_trials(
    OrderedDict([
        ("Online", online_reacher),
        ("Offline", offline_reacher),
    ]),
    y_keys="Final  distance Mean",
    x_key="Number of env steps total",
    process_time_series=plot.padded_ma_filter(10, avg_only_from_left=True),
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.title("Visual Reacher Online Ablation")
plt.legend(["Online", "Offline"], bbox_to_anchor=(0.49, -0.2), loc="upper center", ncol=4, handlelength=1)
plt.tight_layout()
plt.savefig(output_dir + "reacher_online_ablation.pdf")
print("File saved to", output_dir + "reacher_online_ablation.pdf")
