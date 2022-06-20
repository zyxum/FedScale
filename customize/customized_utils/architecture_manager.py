import random
import torch
import onnx
import collections
import networkx as nx
from net2net import get_model_layer, widen_conv, set_model_layer

def flatten(outputs):
    new_output = []
    for i in range(len(outputs)):
        if isinstance(outputs[i], list):
            new_output += flatten(outputs[i])
        else:
            new_output.append(outputs[i])
    return new_output

class Architecture_Manager(object):
    def __init__(self, dummy_input, model_path) -> None:
        self.model_path = model_path
        self.dummy_input = dummy_input
        
    def export_model(self, torch_model):
        torch.onnx.export(torch_model, self.dummy_input, self.model_path,
            export_params=True, verbose=0, training=1, do_constant_folding=False)

    def parse_model(self, torch_model):
        self.export_model(torch_model)
        self.construct_dag()

    def construct_dag(self):
        # load onnx model
        onnx_model = onnx.load(self.model_path)
        graph = onnx_model.graph
        nodes = [node for node in onnx_model.graph.node]
        # get shapes of nodes
        init = graph.initializer
        nodes_shape = {}
        param_names = set()
        for w in init:
            nodes_shape[w.name] = w.dims
            param_names.add(w.name)
        self.dag = nx.DiGraph(name='test')
        # add nodes to graph
        node_inputs = collections.defaultdict(list)
        node_param_shapes = collections.defaultdict(list)
        node_outputs = collections.defaultdict(list)
        node_names = collections.defaultdict(list)
        output2id = {}
        self.trainable_name2id = {}
        self.trainable_layers = []
        self.id2trainable_name = {}
        # construct nodes
        for idx, node in enumerate(nodes):
            for input_name in node.input:
                if input_name not in param_names:
                    node_inputs[idx].append(input_name)
                else:
                    node_param_shapes[idx].append(nodes_shape[input_name])
                    node_names[idx].append(input_name)
            for output_name in node.output:
                node_outputs[idx].append(output_name)
                output2id[output_name] = idx
            node_attr = {
                'param_shapes': node_param_shapes[idx],
                'name': node_names[idx],
                'inputs': node_inputs[idx],
                'outputs': node_outputs[idx]
            }
            self.dag.add_node(idx, attr=node_attr)
            if len(node_param_shapes[idx]) != 0 :
                trainable_name = node_names[idx][0].split('.')[:-1]
                trainable_name = ".".join(trainable_name)
                if len(node_param_shapes[idx]) != 4:
                    self.trainable_layers.append(trainable_name)
                self.trainable_name2id[trainable_name] = idx
                self.id2trainable_name[idx] = trainable_name
        self.entry_idx = None
        for idx in list(self.dag.nodes):
            for input_name in self.dag.nodes[idx]['attr']['inputs']:
                if input_name in output2id.keys():
                    self.dag.add_edge(output2id[input_name], idx)
                else:
                    self.entry_idx = idx
    
    def get_trainable_layer_names(self):
        return self.trainable_layers

    def get_trainable_layer_shape(self, idx):
        return self.dag.nodes[idx]['attr']['param_shapes']

    def trainable_layer_name2node_id(self, name):
        return self.trainable_name2id[name]
    
    def query_trainable_desc_helper(self, node_id):
        outputs = []
        for descendant in self.dag.neighbors(node_id):
            outputs.append(descendant)
        for i in range(len(outputs)):
            if 'Add' in self.dag.nodes[outputs[i]]['attr']['outputs'][0] or\
                'Add' in self.dag.nodes[outputs[i]]['attr']['inputs'][0]:
                raise Exception("not support widen before add")
            if len(self.dag.nodes[outputs[i]]['attr']['param_shapes']) > 0:
                outputs.append(outputs[i])
            if len(self.dag.nodes[outputs[i]]['attr']['param_shapes']) != 1:
                outputs[i] = self.query_trainable_desc_helper(outputs[i])
        outputs = list(set(flatten(outputs)))
        return outputs

    def query_trainable_desc(self, node_id):
        outputs = self.query_trainable_desc_helper(node_id)
        return [self.id2trainable_name[out] for out in outputs]

    def widen(self, torch_model, layer_name, ratio: float=2, noise_factor: float=5e-2):
        if layer_name == 'fc':
            raise Exception("not support widen fc layer")
        parent_id = self.trainable_layer_name2node_id(layer_name)
        children_layers = self.query_trainable_desc(parent_id)
        children_bns = []
        children_convs = []
        for child_layer in children_layers:
            child_id = self.trainable_layer_name2node_id(child_layer)
            child_shape = self.get_trainable_layer_shape(child_id)
            if len(child_shape) == 4:
                children_bns.append(child_layer)
            else:
                children_convs.append(child_layer)
        
        # extract layers and params
        children_bn_layers = [get_model_layer(torch_model, child_layer) for child_layer in children_bns]
        children_conv_layers = [get_model_layer(torch_model, child_layer) for child_layer in children_convs]
        parent_conv_layer = get_model_layer(torch_model, layer_name)
        children_bn_params = [children_bn_layer.state_dict() for children_bn_layer in children_bn_layers]
        children_conv_params = [children_conv_layer.state_dict() for children_conv_layer in children_conv_layers]
        parent_conv_params = parent_conv_layer.state_dict()
        
        # generate mapping randomly
        original_out_channel = parent_conv_params['weight'].shape[0]
        new_out_channel = int(ratio * original_out_channel)
        assert(original_out_channel < new_out_channel)
        self.mapping = list(range(original_out_channel))
        extra_mapping = random.choices(self.mapping, k=new_out_channel-original_out_channel)
        self.mapping += extra_mapping

        # doing net2net widen
        new_parent_conv_params, new_children_params, new_bns_params =\
            widen_conv(parent_conv_params, children_conv_params, self.mapping, children_bn_params, noise_factor=noise_factor)
        
        # create replacement layers
        new_parent_layer = torch.nn.Conv2d(
            parent_conv_layer.in_channels, 
            new_out_channel,
            parent_conv_layer.kernel_size,
            stride=parent_conv_layer.stride,
            padding=parent_conv_layer.padding,
            groups=parent_conv_layer.groups,
            bias=True if parent_conv_layer.bias is not None else False
            )
        new_parent_layer.load_state_dict(new_parent_conv_params)
        new_children_conv_layers = []
        for child_conv_layer, new_child_param in zip(children_conv_layers, new_children_params):
            layer = torch.nn.Conv2d(
                new_out_channel,
                child_conv_layer.out_channels,
                child_conv_layer.kernel_size,
                stride=child_conv_layer.stride,
                padding=child_conv_layer.padding,
                groups=child_conv_layer.groups,
                bias=True if child_conv_layer.bias is not None else False
            )
            layer.load_state_dict(new_child_param)
            new_children_conv_layers.append(layer)
        new_children_bn_layers = []
        for child_bn_layer, new_child_param in zip(children_bn_layers, new_bns_params):
            layer = torch.nn.BatchNorm2d(
                num_features=new_out_channel,
                eps=child_bn_layer.eps,
                momentum=child_bn_layer.momentum,
                affine=child_bn_layer.affine,
                track_running_stats=child_bn_layer.track_running_stats
            )
            layer.load_state_dict(new_child_param)
            new_children_bn_layers.append(layer)
        
        # load new layers to the model
        set_model_layer(torch_model, new_parent_layer, layer_name)
        for child_conv_layer_name, new_child_conv_layer in zip(children_convs, new_children_conv_layers):
            set_model_layer(torch_model, new_child_conv_layer, child_conv_layer_name)
        for child_bn_layer_name, new_child_bn_layer in zip(children_bns, new_children_bn_layers):
            set_model_layer(torch_model, new_child_bn_layer, child_bn_layer_name)
        return torch_model
