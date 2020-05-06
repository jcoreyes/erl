import torch
import sys
import pickle

if __name__ == "__main__":
    filename = sys.argv[1]
    model = pickle.load(open(filename, "rb"))
    torch.save(model, filename + ".pt")
