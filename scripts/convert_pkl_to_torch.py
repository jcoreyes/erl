import pickle
import torch

path = "/checkpoint/anair17/ashvin/rfeatures/multitask1/run2/id0/itr_0"
model = pickle.load(open(path + ".pkl", "rb"))
torch.save(model.state_dict(), path + ".pt")

print(path + ".pt")


