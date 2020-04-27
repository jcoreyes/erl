import pickle
import glob
import numpy as np

def print_stats(data):
    returns = []
    path_lengths = []

    print("num trajectories", len(data))

    for path in data:
        rewards = path["rewards"]
        returns.append(np.sum(rewards))
        path_lengths.append(len(rewards))

    print("returns")
    print("min", np.min(returns))
    print("max", np.max(returns))
    print("mean", np.mean(returns))
    print("std", np.std(returns))

    print("path lengths")
    print("min", np.min(path_lengths))
    print("max", np.max(path_lengths))
    print("mean", np.mean(path_lengths))
    print("std", np.std(path_lengths))

# input_patterns = [
#     "/home/ashvin/data/s3doodad/ashvin/icml2020/hand/pen/demo-bc1/run5/id0/video_*.p",
# ]
# output_file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc1.npy"

# input_patterns = [
#     "/home/ashvin/data/s3doodad/ashvin/icml2020/hand/pen/demo-bc5/run0/id*/video_*.p",
# ]
# output_file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc2.npy"


# input_patterns = [
#     "/home/ashvin/data/s3doodad/ashvin/icml2020/hand/pen/demo-bc5/run4/id*/video_*.p",
# ]
# output_file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc3.npy"

# input_patterns = [
#     "/home/ashvin/data/s3doodad/ashvin/icml2020/hand/pen/demo-bc5/run4/id*/video_*vae.p",
# ]
# output_file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc3_vae.npy"

# input_patterns = [
#     "/home/ashvin/data/s3doodad/ashvin/icml2020/hand/pen/demo-bc5/run4/id*/video_*env.p",
# ]
# output_file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc3_env.npy"

# input_patterns = [
#     "/home/ashvin/data/s3doodad/ashvin/icml2020/hand/pen/demo-bc5/run6/id*/video_*vae.p",
# ]
# output_file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc4_vae.npy"

# input_patterns = [
#     "/home/ashvin/data/s3doodad/ashvin/icml2020/hand/door/demo-bc5/run2/id*/video_*.p",
# ]
# output_file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/door_bc1.npy"

# input_patterns = [
#     "/home/ashvin/data/s3doodad/ashvin/icml2020/hand/hammer/demo-bc1/run0/id*/video_*.p",
# ]
# output_file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/hammer_bc1.npy"

# input_patterns = [
#     "/home/ashvin/data/s3doodad/ashvin/icml2020/hand/relocate/demo-bc1/run0/id*/video_*.p",
# ]
# output_file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/relocate_bc1.npy"

# input_patterns = [
#     "/home/ashvin/data/s3doodad/ashvin/icml2020/hand/door/bc/bc-data1/run0/id*/video_*.p",
# ]
# output_file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/door_bc2.npy"

input_patterns = [
    "/media/ashvin/data2/s3doodad/ashvin/rfeatures/rlbench/open-drawer-vision3/td3bc-with-state3/run0/id0/video_*_vae.p",
]
output_file = "/home/ashvin/data/s3doodad/demos/icml2020/rlbench/rlbench_bc1.npy"

data = []
for pattern in input_patterns:
    for file in glob.glob(pattern):
        d = pickle.load(open(file, "rb"))
        print(file, len(d))
        for path in d: # for deleting image observations
            for i in range(len(path["observations"])):
                ob = path["observations"][i]
                keys = list(ob.keys())
                for key in keys:
                    if key != "state_observation":
                        del ob[key]
        data.extend(d)

pickle.dump(data, open(output_file, "wb"))
print(output_file)
print_stats(data)
