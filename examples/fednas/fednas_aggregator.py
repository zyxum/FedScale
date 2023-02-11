from fedscale.cloud.aggregation.aggregator import Aggregator
from darts.model_search import Network
from darts.model import NetworkCIFAR
import fednas_config_parser as parser
from fedscale.cloud.fllibs import outputClass
import fedscale.cloud.commons as commons

import torch.nn as nn
from torchinfo import summary
import numpy as np
import torch
import numpy
import logging


class FedNAS_Aggregator(Aggregator):
    def __init__(self, args):
        super().__init__(args)

        self.all_trained_size = 0
        self.stage = "search"

    def init_model(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        model = Network(self.args.init_channels, num_classes=outputClass[self.args.data_set],
                        layers=self.args.layers, criterion=criterion, device=self.device)
        self.model = model
        self.model_weights = model.state_dict()
        self.model_arch = model.arch_parameters()
        summary(self.model, input_size=(10, 3, 28, 28))


    def get_global_model(self):
        if self.round >= self.args.search_round:
            return self.model
        return {
            "model_params": self.model.state_dict(),
            "arch_params": self.model.arch_parameters()
        }

    def aggregate_client_arch(self, trained_size, local_arch):
        for p in local_arch:
            if self.model_in_update == 1:
                self.model_arch[p].data = local_arch[p] * float(trained_size)
            else:
                self.model_arch[p].data += local_arch[p] * float(trained_size)

        if self.model_in_update == self.tasks_round:
            for p in self.model_arch:
                d_type = self.model_arch[p].data.dtype

                self.model_arch[p].data = (
                    self.model_arch[p]/float(self.all_trained_size)).to(dtype=d_type)

    def aggregate_client_weights(self, results):
        trained_size = results["trained_size"]
        if self.model_in_update == 1:
            self.all_trained_size = trained_size
        else:
            self.all_trained_size += trained_size

        for p in results['update_weight']:
            param_weight = results['update_weight'][p]
            if isinstance(param_weight, list):
                param_weight = np.asarray(param_weight, dtype=np.float32)
            param_weight = torch.from_numpy(
                param_weight).to(device=self.device)

            if self.model_in_update == 1:
                self.model_weights[p].data = param_weight * float(trained_size)
            else:
                self.model_weights[p].data += param_weight * float(trained_size)

        if self.model_in_update == self.tasks_round:
            for p in self.model_weights:
                d_type = self.model_weights[p].data.dtype

                self.model_weights[p].data = (
                    self.model_weights[p]/float(self.all_trained_size)).to(dtype=d_type)

        if self.stage == "search":
            self.aggregate_client_arch(trained_size, results["arch"])

    def round_weight_handler(self, last_model):
        self.model.load_state_dict(self.model_weights)
        if self.stage == "search":
            for a_g, model_arch in zip(self.model_arch, self.model.arch_parameters()):
                model_arch.data.copy_(a_g.data)

    def round_completion_handler(self):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
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
        self.flatten_client_duration = numpy.array(flatten_client_duration)
        self.round_duration = round_duration
        self.model_in_update = 0
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []
        self.loss_accumulator = []
        self.update_default_task_config()

        if self.round == self.args.search_round:
            self.stage = "train"
            self.finalize_model()

        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0 and self.stage == "train":
            # disable testing when searching
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.MODEL_TEST)
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.START_ROUND)

    def finalize_model(self):
        genotype = self.model.genotype()
        self.model = NetworkCIFAR(self.args.init_channels, outputClass[self.args.data_set],
                                  self.args.layers, self.args.auxiliary, genotype)
        summary(self.model, input_size=(10, 3, 28, 28))



if __name__ == "__main__":
    aggregator = FedNAS_Aggregator(parser.args)
    aggregator.run()