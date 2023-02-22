from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud import commons
from fedscale.cloud.fllibs import outputClass, init_model
import os

import fjord_config_parser as parser
from fjord_utils import sample_subnetwork

import logging
import numpy as np
import numpy
import torch
import pickle


class FjORD_Aggregator(Aggregator):

    def __init__(self, args):
        super(FjORD_Aggregator, self).__init__(args)
        self.sub_models = {}
        self.sub_model_weights = {}
        self.sub_model_count = {}
        self.trained_data_size = .0
        self.client_accuracy = {}
        self.client_best_model = {}
        self.average_test_accuracy = .0
        self.model_to_test = []
        self.test_received = 0
        self.model_accuracy = {}

        # setup hardware
        self.client_tier = {}
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
        logging.info(f"current model partition {self.dp}")

    def init_model(self):
        """Load global model and sample sub models
        """
        assert self.args.engine == commons.PYTORCH

        if self.args.model == "fjord-cnn":
            from fjord_models.fjord_cnn import FjORD_CNN
            self.model = FjORD_CNN(num_classes=outputClass[self.args.data_set])
        elif self.args.model_name != 'None':
            with open(f'/users/yuxuanzh/FedScale/docker/models/{self.args.model_name}.pth.tar', 'rb') as f:
                logging.info(f'loading checkpoint')
                self.model = pickle.load(f)
        else:
            self.model = init_model()


        self.init_submodels()

    def init_submodels(self):
        tier = 0
        for ratio, _ in self.dp:
            submodel = sample_subnetwork(self.model, ratio)
            self.sub_models[tier] = submodel
            self.sub_model_weights[tier] = submodel.state_dict()
            self.sub_model_count[tier] = {}
            tier += 1
        for tier in self.sub_models:
            logging.info(f"log tier {tier} model:")
            logging.info(f"{self.sub_models[tier]}")

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

            probability = [prob for _, prob in self.dp]
            one_hot = np.random.multinomial(1, probability)
            for rank, res in enumerate(one_hot):
                if res == 1:
                    tier = rank
                    break
            # tier = 4
            self.client_tier[clientId] = tier
            # logging.info(f"partition client {clientId} into tier {tier}")
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
        if self.round == self.args.rounds // 2:
            self.args.learning_rate /= 10
        elif self.round == self.args.rounds * 3 // 4:
            self.args.learning_rate /= 10
        conf = {
            'learning_rate': self.args.learning_rate,
            'tier': self.client_tier[clientId],
            'p': self.dp[self.client_tier[clientId]][0],
            'p_pool': [self.dp[tier][0] for tier in range(len(self.dp))],
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
                for tier in self.sub_model_weights:
                    model_weights = self.sub_model_weights[tier]
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

            for tier in self.sub_model_weights:
                model_weights = self.sub_model_weights[tier]
                model_weights_count = self.sub_model_count[tier]
                self.sub_model_weights[tier], self.sub_model_count[tier] = \
                    self.aggregate_weight_helper(param_weight, p, model_weights, model_weights_count, float(results["trained_size"]))

                if self.model_in_update == self.tasks_round:
                    d_type = self.sub_model_weights[tier][p].data.dtype
                    zero_indexes = (self.sub_model_count[tier][p] == 0)
                    self.sub_model_count[tier][p][zero_indexes] = 1.0
                    self.sub_model_weights[tier][p].data = (
                        torch.div(
                            self.sub_model_weights[tier][p].data,
                            self.sub_model_count[tier][p]
                        )
                    ).to(dtype=d_type)


    def aggregate_weight_helper(self, weight: torch.Tensor, weight_name: str, model_weight: dict, 
                                model_weight_count: dict, data_size: float):
        if self.model_in_update == 1:
            model_weight[weight_name] = torch.zeros_like(model_weight[weight_name])
            model_weight_count[weight_name] = torch.zeros_like(model_weight[weight_name])
        if weight.dim() == 0:
            model_weight[weight_name] = weight
        elif weight.dim() == 1:
            dim1 = min(weight.shape[0], model_weight[weight_name].shape[0])
            model_weight[weight_name][:dim1] += weight[:dim1] * data_size
            model_weight_count[weight_name][:dim1] += data_size
        elif weight.dim() == 2:
            dim1 = min(weight.shape[0], model_weight[weight_name].shape[0])
            dim2 = min(weight.shape[1], model_weight[weight_name].shape[1])
            model_weight[weight_name][:dim1, :dim2] += weight[:dim1, :dim2] * data_size
            model_weight_count[weight_name][:dim1, :dim2] += data_size
        elif weight.dim() == 4:
            # logging.info(f"{weight_name}: {model_weight[weight_name].shape}, {weight.shape}")
            dim1 = min(weight.shape[0], model_weight[weight_name].shape[0])
            dim2 = min(weight.shape[1], model_weight[weight_name].shape[1])
            dim3 = min(weight.shape[2], model_weight[weight_name].shape[2])
            dim4 = min(weight.shape[3], model_weight[weight_name].shape[3])
            model_weight[weight_name][:dim1, :dim2, :dim3, :dim4] += weight[:dim1, :dim2, :dim3, :dim4] * data_size
            model_weight_count[weight_name][:dim1, :dim2, :dim3, :dim4] += data_size
        else:
            raise Exception(f"unsupported weight shape: {weight.shape}")
        return model_weight, model_weight_count

    def round_completion_handler(self):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        # logging.info(f"debug check {self.sub_models[0].state_dict()['conv1.weight'][0,0,0,0]}")

        self.global_virtual_clock += self.round_duration
        self.round += 1

        # handle the global update w/ current and last
        self.round_weight_handler(self.last_gradient_weights)

        avgUtilLastround = sum(self.stats_util_accumulator) / \
            max(1, len(self.stats_util_accumulator))
        # assign avg reward to explored, but not ran workers
        for clientId in self.round_stragglers:
            self.client_manager.register_feedback(clientId, avgUtilLastround,
                                              time_stamp=self.round,
                                              duration=self.virtual_client_clock[clientId]['computation'] +
                                              self.virtual_client_clock[clientId]['communication'],
                                              success=False)

        avg_loss = sum(self.loss_accumulator) / \
            max(1, len(self.loss_accumulator))
        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round}, Planned participants: " +
                     f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        # update select participants
        self.sampled_participants = self.select_participants(
            select_num_participants=self.args.num_participants, overcommitment=self.args.overcommitment)
        (clientsToRun, round_stragglers, virtual_client_clock, round_duration, flatten_client_duration) = self.tictak_client_tasks(
            self.sampled_participants, self.args.num_participants)

        logging.info(f"Selected participants to run: {clientsToRun}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)
        self.tasks_round = len(clientsToRun)

        # Update executors and participants
        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(
                self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id)
                                      for c_id in self.sampled_participants]

        self.save_last_param()
        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.flatten_client_duration = np.array(flatten_client_duration)
        self.round_duration = round_duration
        self.model_in_update = 0
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []
        self.loss_accumulator = []
        self.update_default_task_config()

        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0:
            self.test_received = 0
            self.test_result_accumulator = []
            self.model_to_test = list(range(len(self.sub_models)))
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.MODEL_TEST)
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.START_ROUND)


if __name__ == "__main__":
    aggregator = FjORD_Aggregator(parser.args)
    aggregator.run()

