import matplotlib
from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    format_func,
    our_method_name,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from railrl.visualization import plot_util as plot

configure_matplotlib(matplotlib)


dirs = [
          ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-multi3/run1',
       ]
f = plot.filter_by_flat_params({'algo_kwargs.num_updates_per_env_step': 4, 'rdim': 16, 'replay_kwargs.fraction_goals_are_rollout_goals': 0.2})
her = plot.load_exps(dirs, f, suppress_output=True)

dirs = [
          ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-multi3/run1',
       ]
f = plot.filter_by_flat_params({'algo_kwargs.num_updates_per_env_step': 4, 'rdim': 16, 'replay_kwargs.fraction_goals_are_rollout_goals': 1.0, 'replay_kwargs.fraction_goals_are_env_goals': 0.0})
norelabel = plot.load_exps(dirs, f, suppress_output=True)

dirs = [
     ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-multi3-fullrelabel/run1',
]
fullrelabel = plot.load_exps(dirs, suppress_output=True)
plot.comparison(her + fullrelabel + norelabel, "Final  total_distance Mean",
           ["replay_kwargs.fraction_goals_are_rollout_goals", "replay_kwargs.fraction_goals_are_env_goals", ],
#            ["training_mode", "replay_kwargs.fraction_goals_are_env_goals", "replay_kwargs.fraction_goals_are_rollout_goals", "rdim"],
           default_vary={"replay_strategy": "future"},
          smooth=plot.padded_ma_filter(10), figsize=(7.5, 4),
          xlim=(0, 500000), ylim=(0.15, 0.35),
                method_order=[1, 2, 0, 3])
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.ylabel("")
plt.xlabel("Timesteps")
plt.title("Visual Multi-object Pusher")
leg = plt.legend([our_method_name, "None", "Future", "VAE", ],
    bbox_to_anchor=(1.0, 0.5), loc="center left",)
# leg.get_frame().set_alpha(0.9)
plt.tight_layout()
plt.savefig(output_dir + "multiobj_pusher_relabeling_ablation.pdf")
print("File saved to", output_dir + "multiobj_pusher_relabeling_ablation.pdf")
