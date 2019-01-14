import matplotlib.pyplot as plt
from railrl.misc.np_util import truncated_geometric

truncated_geom_factor = 1
max_tau = 5
batch_size = 10000

num_steps_left = truncated_geometric(
    p=truncated_geom_factor / max_tau,
    truncate_threshold=max_tau,
    size=(batch_size, 1),
    new_value=0,
)
print(num_steps_left.max())

a = plt.hist(num_steps_left, bins=max_tau)
print(a)
plt.show()
