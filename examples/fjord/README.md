# Implementation of FjORD on FedScale
> Horvath, S., Laskaridis, S., Almeida, M., Leontiadis, I., Venieris, S., & Lane, N. (2021). Fjord: Fair and accurate federated learning under heterogeneous targets with ordered dropout. Advances in Neural Information Processing Systems, 34, 12876-12889.

https://arxiv.org/abs/2102.13451

## Experiment setting
_referring to section 5_
- uniform-5 setup
  - saying drop rates are [0.2, 0.4, 0.6, 0.8, 1]
- drop scale = 1.0
  - saying probability = [0.2, 0.2, 0.2, 0.2, 0.2]


## Hyperparameters
_referring to section C.3.2 of the paper_
- 10 clients per round
- 1 local epoch per client per round
- 500 rounds
- SGD w/o momentum
- divide client step size by 10 at 50% and 75% of total rounds
- ResNet18:
  - local batch size = 32
  - step size = 0.1
- CNN:
  - local batch size = 16
  - step size = 0.1
- RNN:
  - local batch size = 4
  - step size = 1.0
  
(guess: "step size" is learning rate)