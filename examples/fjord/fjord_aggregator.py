from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud import commons
from fedscale.cloud.fllibs import outputClass, init_model

from fjord_utils import sample_subnetwork

import logging
import numpy as np
import torch
import copy

class FjORD_Aggregator(Aggregator):

    def __init__(self, args):
        super(FjORD_Aggregator, self).__init__(args)
        self.sub_models = {}
        self.sub_model_weights = {}
        self.trained_data_size = 0

        # setup hardware
        if self.args.model_zoo == "fjord-paper":
            self.client_tier = {}
            self.eval_mode = "repr"
            uniform = self.args.uniform
            drop_scale = self.args.drop_scale
            self.dp = []
            ratio = 0
            ratio_incre = 1 / uniform
            dis_accum = .0
            for tier in range(uniform):
                ratio += ratio_incre
                if tier != uniform - 1:
                    self.dp.append([ratio, drop_scale / uniform])
                    dis_accum += drop_scale / uniform
                else:
                    self.dp.append([ratio, 1 - dis_accum])
        else:
            self.eval_mode = "eval"

    def init_model(self):
        """Load global model and sample sub models
        """
        assert self.args.engin == commons.PYTORCH

        if self.args.model_zoo == "fjord-paper":
            if self.args.model == "resnet18":
                from fjord_models.fjord_resnet18 import get_resnet18
                self.model = get_resnet18(num_classes=outputClass[self.args.data_set])
            elif self.args.model == "cnn":
                from fjord_models.fjord_cnn import FjORD_CNN
                self.model = FjORD_CNN(num_classes=outputClass[self.args.data_set])
            else:
                raise ValueError(f"Not support {self.args.model} from FjORD paper")
        else:
            self.model = init_model()

        self.init_submodels()

    def init_submodels(self):
        tier = 0
        for ratio, _ in self.dp:
            submodel = sample_subnetwork(self.model, ratio)
            self.sub_models[tier] = submodel
            self.sub_model_weights[ratio] = submodel.state_dict()
            tier += 1

    def client_register_handler(self, executorId, info):
        """Triggered once receive new executor registration.

                Args:
                    executorId (int): Executor Id
                    info (dictionary): Executor information

                """
        logging.info(f"Loading {len(info['size'])} client traces ...")
        for _size in info['size']:
            # since the worker rankId starts from 1, we also configure the initial dataId as 1
            mapped_id = (self.num_of_clients + 1) % len(
                self.client_profiles) if len(self.client_profiles) > 0 else 1
            systemProfile = self.client_profiles.get(
                mapped_id, {'computation': 1.0, 'communication': 1.0})

            clientId = (
                    self.num_of_clients + 1) if self.experiment_mode == commons.SIMULATION_MODE else executorId

            if self.args.model_zoo == "fjord_paper":
                probability = [prob for _, prob in self.dp]
                one_hot = np.random.multinomial(1, probability)
                for rank, res in enumerate(one_hot):
                    if res == 1:
                        tier = rank
                self.client_tier[clientId] = rank
            self.client_manager.register_client(
                executorId, clientId, size=_size, speed=systemProfile)
            self.client_manager.registerDuration(
                clientId,
                batch_size=self.args.batch_size,
                local_steps=self.args.local_steps,
                upload_size=self.model_update_size,
                download_size=self.model_update_size
            )
            self.num_of_clients += 1

        logging.info("Info of all feasible clients {}".format(
            self.client_manager.getDataInfo()))

    def get_client_conf(self, clientId):
        """Training configurations that will be applied on clients,
        developers can further define personalized client config here.

        Args:
            clientId (int): The client id.

        Returns:
            dictionary: Client training config.

        """
        conf = {
            'learning_rate': self.args.learning_rate,
            'p': self.client_tier[clientId],
            'p_pool': [self.dp[tier][0] for tier in self.dp],
        }
        return conf

    def get_global_model(self):
        """Get global model that would be used by all FL clients (in default FL)

        Returns:
            PyTorch or TensorFlow module: Based on the executor's machine learning framework, initialize and return the model for training.

        """
        return self.sub_models

    def round_weight_handler(self, last_model):
        """Update model when the round completes

        Args:
            last_model (list): A list of global model weight in last round.

        """
        if self.round > 1:
            if self.args.engine == commons.TENSORFLOW:
                for layer in self.model.layers:
                    layer.set_weights([p.cpu().detach().numpy()
                                      for p in self.model_weights[layer.name]])
            else:
                for tier, model_weights in self.sub_model_weights:
                    self.sub_models[tier].load_state_dict(model_weights)
                current_grad_weights = [param.data.clone()
                                        for param in self.model.parameters()]
                self.optimizer.update_round_gradient(
                    last_model, current_grad_weights, self.model)


    def aggregate_client_weights(self, results):
        for p in results["update_weight"]:
            param_weight = results["update_weight"][p]
            if isinstance(param_weight, list):
                param_weight = np.asarray(param_weight, dtype=np.float32)
            param_weight = torch.from_numpy(
                param_weight).to(device=self.device)

            for _, model_weights in self.sub_model_weights:
                self.aggregate_weight_helper(param_weight, p, model_weights, results["trained_size"])

                if self.model_in_update == self.tasks_round:
                    for p in model_weights:
                        for idx in range(len(self.model_weights[p])):
                            d_type = self.model_weights[p][idx].data.dtype

                            self.model_weights[p][idx].data = (
                                    self.model_weights[p][idx].data / float(self.trained_data_size)
                            ).to(dtype=d_type)
        if self.model_in_update == self.tasks_round:
            self.trained_data_size = 0


    def aggregate_weight_helper(self, weight: torch.Tensor, weight_name: str, model_weight, data_size: int):
        if weight.dim() == 0:
            model_weight[weight_name] = weight
        elif weight.dim() == 1:
            dim1 = min(weight.shape[0], model_weight[weight_name].shape[0])
            if weight.shape[0] <= model_weight[weight_name].shape[0]:
                if self.model_in_update == 0:
                    model_weight[weight_name][:dim1] = weight * data_size
                else:
                    model_weight[weight_name][:dim1] += weight * data_size
            else:
                if self.model_in_update == 0:
                    model_weight[weight_name] = weight[:dim1] * data_size
                else:
                    model_weight[weight_name] += weight[:dim1] * data_size
        elif weight.dim() == 2:
            dim1 = min(weight.shape[0], model_weight[weight_name].shape[0])
            dim2 = min(weight.shape[1], model_weight[weight_name].shape[1])
            if weight.shape[0] <= model_weight[weight_name].shape[0]:
                if self.model_in_update == 0:
                    model_weight[weight_name][:dim1, :dim2] = weight * data_size
                else:
                    model_weight[weight_name][:dim1, :dim2] += weight * data_size
            else:
                if self.model_in_update == 0:
                    model_weight[weight_name] = weight[:dim1, :dim2] * data_size
                else:
                    model_weight[weight_name] += weight[:dim1, :dim2] * data_size
        elif weight.dim() == 4:
            dim1 = min(weight.shape[0], model_weight[weight_name].shape[0])
            dim2 = min(weight.shape[1], model_weight[weight_name].shape[1])
            dim3 = min(weight.shape[2], model_weight[weight_name].shape[2])
            dim4 = min(weight.shape[3], model_weight[weight_name].shape[3])
            if weight.shape[0] <= model_weight[weight_name].shape[0]:
                if self.model_in_update == 0:
                    model_weight[weight_name][:dim1, :dim2, :dim3, :dim4] = weight * data_size
                else:
                    model_weight[weight_name][:dim1, :dim2, :dim3, :dim4] += weight * data_size
            else:
                if self.model_in_update == 0:
                    model_weight[weight_name] = weight[:dim1, :dim2, :dim3, :dim4] * data_size
                else:
                    model_weight[weight_name] += weight[:dim1, :dim2, :dim3, :dim4] * data_size
        else:
            raise Exception(f"unsupported weight shape: {weight.shape}")


