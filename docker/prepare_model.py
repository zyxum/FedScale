import torch.nn as nn
import pickle
import sys


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


model_path = sys.argv[1]

with open(model_path, 'rb') as f:
    model = pickle.load(f)
model.apply(weight_reset)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
