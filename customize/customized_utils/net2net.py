from typing import OrderedDict
import numpy as np
import torch
import collections

def widen_parnet_conv(params: OrderedDict, mapping, noise_factor=5e-2):
    """
    weights have shape: (out_channels, in_channels, kernel_height, kernel_width)
    bisd hsbr shape: (out_channels, )
    """
    new_params = collections.OrderedDict()
    weights = params['weight'].numpy()
    _, in_channel, kernel_height, kernel_width = weights.shape
    new_out_channel = len(mapping)
    new_weights = np.zeros((new_out_channel, in_channel, kernel_height, kernel_width))
    for i in range(len(mapping)):
        new_weights[i, :, :, :] = weights[mapping[i], :, :, :].copy() \
            + np.random.normal(scale=noise_factor*weights[mapping[i], :, :, :].std(), \
            size=list(weights[mapping[i], :, :, :].shape))
    new_params['weight'] = torch.from_numpy(new_weights)
    if 'bias' in params.keys():
        bias = params['bias'].numpy()
        new_bias = np.zeros((new_out_channel))
        for i in range(len(mapping)):
            new_bias[i] = bias[mapping[i]].copy() \
                + np.random.normal(scale=noise_factor, size=list(bias[mapping[i]].shape))
        new_params['bias'] = torch.from_numpy(new_bias)
    return new_params

def widen_child_conv(params: OrderedDict, mapping, noise_factor=5e-2):
    """
    weights have shape: (out_channels, in_channels, kernel_height, kernel_width)
    bisd hsbr shape: (out_channels, )
    """
    new_params = collections.OrderedDict()
    weights = params['weight'].numpy()
    out_channel, _, kernel_height, kernel_width = weights.shape
    new_in_channel = len(mapping)
    new_weights = np.zeros((out_channel, new_in_channel, kernel_height, kernel_width))
    scale = [mapping.count(m) for m in mapping]
    for i in range(len(mapping)):
        new_weights[:, i, :, :] = weights[:, mapping[i], :, :] / scale[i]
    new_params['weight'] = torch.from_numpy(new_weights)
    if 'bias' in params.keys():
        bias = params['bias'].numpy()
        new_bias = np.zeros((new_in_channel,))
        for i in range(len(mapping)):
            new_bias[i] = bias[mapping[i]] / scale[i]
        new_params['bias'] = torch.from_numpy(new_bias)
    return new_params

def widen_batch(batch: OrderedDict, mapping, noise_factor=5e-2):
    new_batch = collections.OrderedDict()
    for param_name in batch.keys():
        batch[param_name] = batch[param_name].numpy()
        if param_name in ['num_batches_tracked']:
            new_batch[param_name] = torch.from_numpy(batch[param_name])
        else:
            new_batch[param_name] = np.zeros((len(mapping), ))
            for i in range(len(mapping)):
                new_batch[param_name][i] = \
                    batch[param_name][mapping[i]].copy() + np.random.normal(scale=noise_factor, size=list(batch[param_name][mapping[i]].shape))
            new_batch[param_name] = torch.from_numpy(new_batch[param_name])
    return new_batch


def widen_conv(parent_params: OrderedDict, children_params: OrderedDict, mapping, bns_params: list=[], noise_factor=5e-2):
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
    new_parent_params = widen_parnet_conv(parent_params, mapping, noise_factor=noise_factor)
    new_children_params = []
    for child_params in children_params:
        new_child_params = widen_child_conv(child_params, mapping, noise_factor=noise_factor)
        new_children_params.append(new_child_params)
    new_bns_params = []
    for bn_params in bns_params:
        new_bn_params = widen_batch(bn_params, mapping, noise_factor)
        new_bns_params.append(new_bn_params)
    return new_parent_params, new_children_params, new_bns_params


def get_model_layer(torch_model, attri_name: str):
    if '.' in attri_name:
        attri_names = attri_name.split('.')
    else:
        attri_names = [attri_name]
    module = torch_model
    for attri in attri_names:
        module = getattr(module, attri)
    return module


def set_model_layer(torch_model, torch_module, attri_name: str):
    if '.' in attri_name:
        attri_names = attri_name.split('.')
    else:
        attri_names = [attri_name]
    to_modify = attri_names[-1]
    attri_name = ".".join(attri_names[:-1])
    module = get_model_layer(torch_model, attri_name)
    setattr(module, to_modify, torch_module)
