import torch
import math
from copy import deepcopy


def get_layer(module: torch.nn.Module, layer_name: str):
    if '.' in layer_name:
        module_name_list = layer_name.split('.')
    else:
        module_name_list = [layer_name]
    for module_name in module_name_list:
        module = getattr(module, module_name)
    return module


def set_layer(layer: torch.nn.Module, module: torch.nn.Module, layer_name: str):
    if '.' in layer_name:
        module_name_list = layer_name.split('.')
    else:
        module_name_list = [layer_name]
    # to_modify = module_name_list[-1]
    for idx, module_name in enumerate(module_name_list):
        if idx == len(module_name_list) - 1:
            setattr(module, module_name, layer)
        else:
            module = getattr(module, module_name)


def shrink_width(old_layer: torch.nn.Module,
                 p: float = 1.0,
                 is_first: bool = False,
                 is_last: bool = False) -> torch.nn.Module:
    if isinstance(old_layer, torch.nn.Conv2d):
        new_in_channels = math.ceil(old_layer.in_channels * p) if not is_first else old_layer.in_channels
        new_out_channels = math.ceil(old_layer.out_channels * p)
        new_layer = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=new_out_channels,
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            dilation=old_layer.dilation,
            groups=old_layer.groups,
            bias=True if old_layer.bias is not None else False
        )
        new_layer.state_dict()["weight"] = old_layer.state_dict()["weight"][:new_in_channels, :new_out_channels, :, :]
        if "bias" in new_layer.state_dict():
            new_layer.state_dict()["bias"] = old_layer.state_dict()["bias"][:new_out_channels]
    elif isinstance(old_layer, torch.nn.Linear):
        new_in_features = math.ceil(old_layer.in_features * p)
        new_out_features = math.ceil(old_layer.out_features * p) if not is_last else old_layer.out_features
        new_layer = torch.nn.Linear(
            in_features=new_in_features,
            out_features=new_out_features,
            bias=True if old_layer.bias is not None else False
        )
        new_layer.state_dict()["weight"] = old_layer.state_dict()["weight"][:new_in_features, :new_out_features]
        if "bias" in new_layer.state_dict():
            new_layer.state_dict()["bias"] = old_layer.state_dict()["bias"][:new_out_features]
    elif isinstance(old_layer, torch.nn.BatchNorm2d):
        new_num_features = math.ceil(old_layer.num_features * p)
        new_layer = torch.nn.BatchNorm2d(
            num_features=new_num_features,
            eps=old_layer.eps,
            momentum=old_layer.momentum,
            affine=old_layer.affine,
            track_running_stats=old_layer.track_running_stats
        )
        new_layer.state_dict()["weight"] = old_layer.state_dict()["weight"][:new_num_features]
        if "bias" in new_layer.state_dict():
            new_layer.state_dict()["bias"] = new_layer.state_dict()["bias"][:new_num_features]
    else:
        raise TypeError(f"{type(old_layer)} not supported in shrinking")
    return new_layer


def sample_subnetwork(model: torch.nn.Module, p: float = 1.0) -> torch.nn.Module:
    # get all layers
    model = deepcopy(model)
    layers = set()
    first = None
    for name, _ in model.named_parameters():
        if "weight" in name or "bias" in name:
            layer_name_split = name.split('.')[:-1]
            layer_name = ".".join(layer_name_split)
            layers.add(layer_name)
            if not first:
                first = layer_name
    last = layer_name
    for layer in layers:
        old_layer = get_layer(model, layer)
        new_layer = shrink_width(old_layer, p, first == layer, last == layer)
        # reset layer
        set_layer(new_layer, model, layer)
    return model


def load_sub_layer(layer: torch.nn.Module, sub_layer: torch.nn.Module):
    if isinstance(sub_layer, torch.nn.Conv2d):
        in_channels = sub_layer.in_channels
        out_channels = sub_layer.out_channels
        layer.state_dict()["weight"][:in_channels, :out_channels, :, :] = deepcopy(
            sub_layer.state_dict()["weight"]
        )
        if "bias" in layer.state_dict():
            layer.state_dict()["bias"][:out_channels] = deepcopy(
                sub_layer.state_dict()["bias"]
            )
    elif isinstance(sub_layer, torch.nn.Linear):
        in_features = sub_layer.in_features
        out_features = sub_layer.out_features
        layer.state_dict()["weight"][:in_features, :out_features] = deepcopy(
            sub_layer.state_dict()["weight"]
        )
        if "bias" in layer.state_dict():
            layer.state_dict()["bias"][:out_features] = deepcopy(
                sub_layer.state_dict()["bias"]
            )
    elif isinstance(sub_layer, torch.nn.BatchNorm2d):
        num_features = sub_layer.num_features
        layer.state_dict()["weight"][:num_features] = deepcopy(
            sub_layer.state_dict()["weight"]
        )
        if "bias" in layer.state_dict():
            layer.state_dict()["bias"][:num_features] = deepcopy(
                sub_layer.state_dict()["bias"]
            )
    else:
        raise TypeError(f"{type(sub_layer)} not supported in shrinking")


def load_sub_model(model: torch.nn.Module, sub_model: torch.nn.Module) -> torch.nn.Module:
    layers = set()
    for name, _ in model.named_parameters():
        if "weight" in name or "bias" in name:
            layer_name_split = name.split('.')[:-1]
            layer_name = ".".join(layer_name_split)
            layers.add(layer_name)
    for layer in layers:
        large_layer = get_layer(model, layer)
        small_later = get_layer(sub_model, layer)
        load_sub_layer(large_layer, small_later)
        set_layer(large_layer, model, layer)
    return model
