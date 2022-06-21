from collections import defaultdict
import logging
import math
import torch
import numpy as np
import os, sys
import pickle
from fedscale.core import events
from fedscale.core.fl_aggregator_libs import logDir
from fedscale.core.aggregator import Aggregator
from fedscale.core.fllibs import init_model
from customized_arg_parser import args
from customized_utils.cluster_manager import Cluster_Manager
from customized_utils.architecture_manager import Architecture_Manager


class Customized_Aggregator(Aggregator):

    def __init__(self, args):
        super().__init__(args)
        self.client_loss_in_update = {}
        self.model_in_update = [0]
        self.global_virtual_clock = [0.]
        self.round_duration = [0.]
        self.round = 0
        self.cluster_manager = Cluster_Manager()

    def run(self):
        self.setup_env()
        self.init_control_communication()
        self.init_data_communication()

        self.init_model()
        self.save_last_param()
        self.model_update_size = sys.getsizeof(pickle.dumps(self.model))/1024.0*8. # kbits
        self.client_profiles = self.load_client_profile(file_path=self.args.device_conf_file)

        dummy_input = torch.randn(10, 3, 32, 32, device=self.device)
        self.archi_manager = Architecture_Manager(dummy_input, 'customize_server.onnx')
        self.archi_manager.parse_model(self.model[-1])

        self.event_monitor()

    def init_model(self):
        assert self.args.engine == events.PYTORCH, "Please define model for non-PyTorch models"
        self.model = [init_model()]
        self.model_weights = [self.model[0].state_dict()]
    
    def save_last_param(self):
        if self.args.engine == events.TENSORFLOW:
            self.last_gradient_weights = [layer.get_weights() for layer in self.model.layers]
        else:
            self.last_gradient_weights = [p.data.clone() for p in self.model[-1].parameters()]

    def get_test_config(self, client_id):
        """FL model testing on clients"""

        return {'model': self.model}

    def testing_completion_handler(self, client_id, results):
        """Each executor will handle a subset of testing dataset"""

        results = results['results']

        # List append is thread-safe
        self.test_result_accumulator.append(results)

        # Have collected all testing results
        if len(self.test_result_accumulator) == len(self.executors):
            accumulator = self.test_result_accumulator[0]
            for i in range(1, len(self.test_result_accumulator)):
                if self.args.task == "detection":
                    for key in accumulator:
                        if key == "boxes":
                            for j in range(self.imdb.num_classes):
                                accumulator[key][j] = accumulator[key][j] + self.test_result_accumulator[i][key][j]
                        else:
                            accumulator[key] += self.test_result_accumulator[i][key]
                else:
                    for key in accumulator:
                        if isinstance(accumulator[key], dict):
                            for subkey in accumulator[key].keys():
                                accumulator[key][subkey] += self.test_result_accumulator[i][key][subkey]
                        else:
                            accumulator[key] += self.test_result_accumulator[i][key]
            if self.args.task == "detection":
                self.testing_history['perf'][self.round] = {'round': self.round, 'clock': self.global_virtual_clock[-1],
                    'top_1': round(accumulator['top_1']*100.0/len(self.test_result_accumulator), 4),
                    'top_5': round(accumulator['top_5']*100.0/len(self.test_result_accumulator), 4),
                    'loss': accumulator['test_loss'],
                    'test_len': accumulator['test_len']
                }
            else:
                self.testing_history['perf'][self.round] = {'round': self.round, 'clock': self.global_virtual_clock[-1],
                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                    'loss': accumulator['test_loss']/accumulator['test_len'],
                    'sp_loss': accumulator['sp_loss'],
                    'test_len': accumulator['test_len']
                }


            logging.info("FL Testing in epoch: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, sploss: {}, test loss: {:.4f}, test len: {}"
                    .format(self.round, self.global_virtual_clock[-1], self.testing_history['perf'][self.round]['top_1'],
                    self.testing_history['perf'][self.round]['top_5'], self.testing_history['perf'][self.round]['sp_loss'],self.testing_history['perf'][self.round]['loss'],
                    self.testing_history['perf'][self.round]['test_len']))

            # Dump the testing result
            with open(os.path.join(logDir, 'testing_perf'), 'wb') as fout:
                pickle.dump(self.testing_history, fout)

            if len(self.loss_accumulator):
                self.log_writer.add_scalar('Test/round_to_loss', self.testing_history['perf'][self.round]['loss'], self.round)
                self.log_writer.add_scalar('Test/round_to_accuracy', self.testing_history['perf'][self.round]['top_1'], self.round)
                self.log_writer.add_scalar('FAR/time_to_test_loss (min)', self.testing_history['perf'][self.round]['loss'],
                                            self.global_virtual_clock[-1]/60.)
                self.log_writer.add_scalar('FAR/time_to_test_accuracy (min)', self.testing_history['perf'][self.round]['top_1'],
                                            self.global_virtual_clock[-1]/60.)
                self.log_writer.add_scalars('Test/sp_loss', self.testing_history['perf'][self.round]['sp_loss'], self.round)

            # widen layer on demand
            if self.need_update:
                model = self.archi_manager.widen(self.testing_history['perf'][self.round]['sp_loss'], self.model[-1])
                self.model.append(model)
                self.need_update = False

            self.broadcast_events_queue.append(events.START_ROUND)


    def client_completion_handler(self, results):
        """We may need to keep all updates from clients, if so, we need to append results to the cache"""
        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility
        #       'val_res': val_res}


        if self.args.gradient_policy in ['q-fedavg']:
            self.client_training_results.append(results)
        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])
        clientId = results['clientId']
        clusterId = self.cluster_manager.query_cluster_id(clientId)
        client_val_res = results['val_res']
        self.client_loss_in_update[clientId] = client_val_res['val_loss']

        self.client_manager.registerScore(results['clientId'], results['utility'],
            auxi=math.sqrt(results['moving_loss']),
            time_stamp=self.round,
            duration=self.virtual_client_clock[results['clientId']]['computation']+
                self.virtual_client_clock[results['clientId']]['communication']
        )

        """
            [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
            H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
        """
        # Start to take the average of updates, and we do not keep updates to save memory
        # Importance of each update is 1/#_of_participants
        # importance = 1./self.tasks_round

        self.update_lock.acquire()
        # ================== Aggregate weights ======================

        self.model_in_update[clusterId] += 1
        if self.model_in_update[clusterId] == 1:
            for p in results['update_weight']:
                temp_list = results['update_weight'][p]
                if isinstance(results['update_weight'][p], list):
                    temp_list = np.asarray(temp_list, dtype=np.float32)
                self.model_weights[clusterId][p].data = torch.from_numpy(temp_list).to(device=self.device)
        else:
            for p in results['update_weight']:
                temp_list = results['update_weight'][p]
                if isinstance(results['update_weight'][p], list):
                    temp_list = np.asarray(temp_list, dtype=np.float32)
                self.model_weights[clusterId][p].data += torch.from_numpy(temp_list).to(device=self.device)
        
        # TODO: split tasks_round for different clusters
        if self.model_in_update[clusterId] == self.tasks_cluster[clusterId]:
            for p in self.model_weights[clusterId]:
                d_type = self.model_weights[clusterId][p].data.dtype
                self.model_weights[clusterId][p].data = (self.model_weights[clusterId][p]/float(self.tasks_cluster[clusterId])).to(dtype=d_type)
        self.update_lock.release()

    def round_weight_handler(self, last_model, clusterId):
        """Update model when the round completes"""
        if self.round > 1:
            if self.args.engine == events.TENSORFLOW:
                for layer in self.model.layers:
                    layer.set_weights([p.cpu().detach().numpy() for p in self.model_weights[layer.name]])
                # TODO: support update round gradient
            else:
                self.model[clusterId].load_state_dict(self.model_weights[clusterId])
                # temporarily disable server optimizer
                # current_grad_weights = [param.data.clone() for param in self.model.parameters()]
                # self.optimizer.update_round_gradient(last_model, current_grad_weights, self.model)


    def select_participants(self, select_num_participants, overcommitment=1.3):
        return sorted(self.client_manager.resampleClients(
            int(select_num_participants*overcommitment), 
            cur_time=self.global_virtual_clock[-1]),
        )

    def round_completion_handler(self):
        self.round += 1
        if self.round % self.args.decay_round == 0:
            self.args.learning_rate = max(self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

        avgUtilLastround = sum(self.stats_util_accumulator)/max(1, len(self.stats_util_accumulator))

        # assign avg reward to explored, but not ran workers
        for clientId in self.round_stragglers:
            self.client_manager.registerScore(clientId, avgUtilLastround,
                    time_stamp=self.round,
                    duration=self.virtual_client_clock['computation']+self.virtual_client_clock[clientId]['communication'],
                    success=False)
        
        avg_loss = sum(self.loss_accumulator)/max(1, len(self.loss_accumulator))
        
        # per cluster update
        for clusterId in self.cluster_manager.clusters.keys():
            self.global_virtual_clock[clusterId] += self.round_duration[clusterId]

            # handle the global update w/ current and last
            self.round_weight_handler(self.last_gradient_weights, clusterId)

            logging.info(f"Cluster: {clusterId}, Wall clock: {round(self.global_virtual_clock[clusterId])} s, round: {self.round}, Planned participants: " + \
                f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        
        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        # update select participants
        self.sampled_participants = self.select_participants(
                        select_num_participants=self.args.total_worker, overcommitment=self.args.overcommitment)
        (clientsToRun, round_stragglers, virtual_client_clock, round_duration, flatten_client_duration) = self.tictak_client_tasks(
                        self.sampled_participants, self.args.total_worker)

        logging.info(f"Selected participants to run: {clientsToRun}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)
        logging.info(f"registered tasks")
        self.tasks_round = len(clientsToRun)

        # create tasks per cluster
        self.tasks_cluster = {}
        for client_id in clientsToRun:
            cluster_id = self.cluster_manager.query_cluster_id(int(client_id))
            if cluster_id not in self.tasks_cluster.keys():
                self.tasks_cluster[cluster_id] = 0
            self.tasks_cluster[cluster_id] += 1
        logging.info(f"recorded tasks for clusters")

        # Update executors and participants
        if self.experiment_mode == events.SIMULATION_MODE:
            self.sampled_executors = list(self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id) for c_id in self.sampled_participants]

        # record validate loss to cluster manager
        if len(self.client_loss_in_update) != 0:
            self.need_update = self.cluster_manager.record_val_loss(self.client_loss_in_update)
        logging.info(f"recorded validate loss")

        self.save_last_param()
        logging.info(f"saved last parameters")

        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.flatten_client_duration = np.array(flatten_client_duration)
        for clusterId in self.cluster_manager.clusters.keys():
            self.round_duration[clusterId] = round_duration
            self.model_in_update[clusterId] = 0
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []
        self.client_loss_in_update = {}

        logging.info(f"finished round completion handler")

        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(events.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0:
            self.broadcast_aggregator_events(events.UPDATE_MODEL)
            self.broadcast_aggregator_events(events.MODEL_TEST)
        else:
            self.broadcast_aggregator_events(events.UPDATE_MODEL)
            self.broadcast_aggregator_events(events.START_ROUND)
        
        logging.info(f"finished broadcast events")

    def get_client_conf(self, clientId):
        """Training configurations that will be applied on clients"""

        conf = {
            'learning_rate': self.args.learning_rate,
            'model': self.model[self.cluster_manager.query_cluster_id(clientId)] # none indicates we are using the global model
        }
        return conf

    def client_register_handler(self, executorId, info):
        """Triggered once receive new executor registration"""

        logging.info(f"Loading {len(info['size'])} client traces ...")
        for _size in info['size']:
            # since the worker rankId starts from 1, we also configure the initial dataId as 1
            mapped_id = (self.num_of_clients+1)%len(self.client_profiles) if len(self.client_profiles) > 0 else 1
            systemProfile = self.client_profiles.get(mapped_id, {'computation': 1.0, 'communication':1.0})

            clientId = (self.num_of_clients+1) if self.experiment_mode == events.SIMULATION_MODE else executorId
            self.cluster_manager.register_client(clientId)
            self.client_manager.registerClient(executorId, clientId, size=_size, speed=systemProfile)
            self.client_manager.registerDuration(clientId, batch_size=self.args.batch_size,
                upload_step=self.args.local_steps, upload_size=self.model_update_size, download_size=self.model_update_size)
            self.num_of_clients += 1

        logging.info("Info of all feasible clients {}".format(self.client_manager.getDataInfo()))

    
    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        if self.experiment_mode == events.SIMULATION_MODE:
            # NOTE: We try to remove dummy events as much as possible in simulations,
            # by removing the stragglers/offline clients in overcommitment"""
            sampledClientsReal = []
            completionTimes = []
            completed_client_clock = {}
            # 1. remove dummy clients that are not available to the end of training
            for client_to_run in sampled_clients:
                client_cfg = self.client_conf.get(client_to_run, self.args)

                exe_cost = self.client_manager.getCompletionTime(client_to_run,
                                        batch_size=client_cfg.batch_size, upload_step=client_cfg.local_steps,
                                        upload_size=self.model_update_size, download_size=self.model_update_size)

                roundDuration = exe_cost['computation'] + exe_cost['communication']
                # if the client is not active by the time of collection, we consider it is lost in this round
                if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock[-1]):
                    sampledClientsReal.append(client_to_run)
                    completionTimes.append(roundDuration)
                    completed_client_clock[client_to_run] = exe_cost

            num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))
            # 2. get the top-k completions to remove stragglers
            sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
            top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            dummy_clients = [sampledClientsReal[k] for k in sortedWorkersByCompletion[num_clients_to_collect:]]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            return (clients_to_run, dummy_clients, 
                    completed_client_clock, round_duration, 
                    completionTimes[:num_clients_to_collect])
        else:
            completed_client_clock = {
                client:{'computation': 1, 'communication':1} for client in sampled_clients}
            completionTimes = [1 for c in sampled_clients]
            return (sampled_clients, sampled_clients, completed_client_clock, 
                1, completionTimes)

    def log_train_result(self, avg_loss):
        """Result will be post on TensorBoard"""
        self.log_writer.add_scalar('Train/round_to_loss', avg_loss, self.round)
        self.log_writer.add_scalar('FAR/time_to_train_loss (min)', avg_loss, self.global_virtual_clock[-1]/60.)
        self.log_writer.add_scalar('FAR/round_duration (min)', self.round_duration[-1]/60., self.round)
        self.log_writer.add_histogram('FAR/client_duration (min)', self.flatten_client_duration, self.round)

    def log_test_result(self):
        self.log_writer.add_scalar('Test/round_to_loss', self.testing_history['perf'][self.round]['loss'], self.round)
        self.log_writer.add_scalar('Test/round_to_accuracy', self.testing_history['perf'][self.round]['top_1'], self.round)
        self.log_writer.add_scalar('FAR/time_to_test_loss (min)', self.testing_history['perf'][self.round]['loss'],
                                    self.global_virtual_clock[-1]/60.)
        self.log_writer.add_scalar('FAR/time_to_test_accuracy (min)', self.testing_history['perf'][self.round]['top_1'],
                                    self.global_virtual_clock[-1]/60.)

if __name__ == "__main__":
    aggregator = Customized_Aggregator(args)
    aggregator.run()
