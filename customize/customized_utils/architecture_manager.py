import torch
import onnx
import collections
import networkx as nx

def flatten(outputs):
    new_output = []
    for i in range(len(outputs)):
        if isinstance(outputs[i], list):
            new_output += flatten(outputs[i])
        else:
            new_output.append(outputs[i])
    return new_output

class Architecture_Manager(object):
    def __init__(self, torch_model, dummy_input, model_path) -> None:
        self.model_path = model_path
        torch.onnx.export(torch_model, dummy_input, self.model_path,
            export_params=True, verbose=0, training=1, do_constant_folding=False)
        
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
            if 'Add' in self.dag.nodes[outputs[i]]['attr']['outputs'][0]:
                print("not support widen before add")
                raise Exception
            if len(self.dag.nodes[outputs[i]]['attr']['param_shapes']) > 0:
                outputs.append(outputs[i])
            if len(self.dag.nodes[outputs[i]]['attr']['param_shapes']) != 1:
                outputs[i] = self.query_trainable_desc_helper(outputs[i])
        outputs = list(set(flatten(outputs)))
        # print(outputs)
        # outputs_names = [self.id2trainable_name[out] for out in outputs]
        return outputs

    def query_trainable_desc(self, node_id):
        outputs = self.query_trainable_desc_helper(node_id)
        return [self.id2trainable_name[out] for out in outputs]