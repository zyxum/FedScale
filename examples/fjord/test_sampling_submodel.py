from fjord_models.fjord_resnet18 import get_resnet18
from fjord_utils import sample_subnetwork

import torch

resnet18 = get_resnet18(num_classes=62)
# print(resnet18.children())
sample_subnetwork(resnet18)
