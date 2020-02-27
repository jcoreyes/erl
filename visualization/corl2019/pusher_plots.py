import matplotlib
import matplotlib.pyplot as plt
from railrl.misc import plot_util as plot
from railrl.misc import data_processing as dp
########## PUSHER PLOT

dirs = [
        '/home/ashvin/data/rail-khazatsky/sasha/cond-rig/hyp-tuning/tuning/batch_size/',
       ]

#f = plot.filter_by_flat_params({'grill_variant.algo_kwargs.batch_size': 128,
#                                'grill_variant.algo_kwargs.num_trains_per_train_loop': 1000})
normal = plot.load_exps(dirs, suppress_output=True, progress_filename="tensorboard_log.csv")

#f = plot.filter_by_flat_params({'grill_variant.algo_kwargs.num_trains_per_train_loop': 4000})
#4000_updates = plot.load_exps(dirs, f, suppress_output=True, progress_filename="tensorboard_log.csv")

#f = plot.filter_by_flat_params({'grill_variant.algo_kwargs.batch_size': 1024})
#batch_1024 = plot.load_exps(dirs, f, suppress_output=True, progress_filename="tensorboard_log.csv")


plot.comparison(
      normal,
      ["evaluation/env_infos/final/current_object_distance_Mean",], # "AverageReturn", "Final puck_distance Mean", "Final hand_and_puck_distance Mean"], 
           # [
           #  'train_vae_variant.algo_kwargs.batch_size',
           #  'train_vae_variant.latent_sizes'
           # ],
           [
            'exp_id',
           ],
           default_vary={"env_kwargs.randomize_position_on_reset": True},
          smooth=plot.padded_ma_filter(100),
           figsize=(8, 4.5),
           xlim=(0, 300),
           #method_order=(0, 2, 1),
          print_final=False, print_min=False, print_plot=True)
# plt.title("Pusher2D, Distance to Goal")
#plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func_epoch))
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.title("Multi-color Pusher Learning Curve")
plt.legend([])
#     ["CC-RIG", "RIG", "Oracle", ],
#     bbox_to_anchor=(1.0, 0.5), loc="center left",
# )
plt.tight_layout()

plt.savefig("/home/ashvin/data/pusher_batch.png")
plt.savefig("/home/ashvin/data/pusher_batch.pdf")