import numpy as np
import onnx
import networkx as nx
import collections

def widen_parnet_conv(weights, bias, mapping, noise_factor=5e-2):
    """
    weights have shape: (out_channels, in_channels, kernel_height, kernel_width)
    bisd hsbr shape: (out_channels, )
    """
    _, in_channel, kernel_height, kernel_width = weights.shape()
    new_out_channel = len(mapping)
    new_weights = np.zeros((new_out_channel, in_channel, kernel_height, kernel_width))
    new_bias = np.zeros((new_out_channel,))
    for i in range(len(mapping)):
        new_weights[i, :, :, :] = weights[mapping[i], :, :, :].copy() \
            + np.random.normal(scale=noise_factor*weights[mapping[i], :, :, :].std(), \
            size=list(weights[mapping[i], :, :, :].shape))
        new_bias[i] = bias[mapping[i]].copy() \
            + np.random.normal(scale=noise_factor, size=list(bias[mapping[i]].shape))
    return new_weights, new_bias

def widen_child_conv(weights, bias, mapping, noise_factor=5e-2):
    """
    weights have shape: (out_channels, in_channels, kernel_height, kernel_width)
    bisd hsbr shape: (out_channels, )
    """    
    out_channel, _, kernel_height, kernel_width = weights.shape()
    new_in_channel = len(mapping)
    new_weights = np.zeros((out_channel, new_in_channel, kernel_height, kernel_width))
    new_bias = np.zeros((new_in_channel,))
    scale = [mapping.count(m) for m in mapping]
    for i in range(len(mapping)):
        new_weights[:, i, :, :] = weights[:, mapping[i], :, :] / scale[i]
        new_bias[i] = bias[mapping[i]] / scale[i]
    return new_weights, new_bias

def widen_batch(weights, bias, mapping, noise_factor=5e-2):
    new_weights = np.zeros((len(mapping), ))
    new_bias = np.zeros((len(mapping), ))
    for i in range(len(mapping)):
        new_weights[i] = weights[mapping[i]] + np.random.normal(scale=noise_factor, size=list(weights[mapping[i]].shape))
        new_bias[i] = bias[mapping[i]] + np.random.normal(scale=noise_factor, size=list(bias[mapping[i]].shape))
    return new_weights, new_bias


def widen_conv(parent_weights, parent_bias, children_weights, children_bias, mapping, batch=None, noise_factor=5e-2):
    """
    parent_weights: weights of a conv layer, shape: (out_channels, in_channels, kernel_height, kernel_width)
    parent_bias: bias of a conv layer, shape: (out_channels,)
    children_weights: weights of all following layers
    children_bias: bias of all following layers
    mapping: instructs how to extend a layer by mapping new kernels to original kernels
    batch: {"weights": , "bias": } batch norm layer between parent and children
    noise_factor: init the new layer with some noise

    return: widened parent_weights, parent_bias, children_weights, and children_bias
    """
    new_parent_weights, new_parent_bias = widen_parnet_conv(parent_weights, parent_bias, mapping, noise_factor=noise_factor)
    new_children_weights = []
    new_children_bias = []
    for child_weights, child_bias in zip(children_weights, children_bias):
        new_child_weights, new_child_bias = widen_child_conv(child_weights, child_bias, mapping, noise_factor=noise_factor)
        new_children_weights.append(new_child_weights)
        new_children_bias.append(new_child_bias)
    if batch is not None:
        new_batch_weights, new_batch_bias = widen_batch(batch['weights'], batch['bias'], mapping, noise_factor=noise_factor)
        new_batch = {
            "weights": new_batch_weights,
            "bias": new_batch_bias
        }
    else:
        new_batch = None
    return new_parent_weights, new_parent_bias, new_children_weights, new_children_bias, new_batch

