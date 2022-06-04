import logging
import math
import torch
import numpy as np
import os
import pickle
from fedscale.core import events
from fedscale.core.fl_aggregator_libs import logDir
from fedscale.core.aggregator import Aggregator
from customized_arg_parser import args
from customized_init_model import customized_init_model

class Customized_Aggregator(Aggregator):

    def __init__(self, args):
        super().__init__(args)
        self.client_loss_accumulator = {}
    
    def init_model(self):
        self.model = customized_init_model()
        self.model_weights = self.model.state_dict()

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
                        accumulator[key] += self.test_result_accumulator[i][key]
            if self.args.task == "detection":
                self.testing_history['perf'][self.round] = {'round': self.round, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']*100.0/len(self.test_result_accumulator), 4),
                    'top_5': round(accumulator['top_5']*100.0/len(self.test_result_accumulator), 4),
                    'loss': accumulator['test_loss'],
                    'test_len': accumulator['test_len']
                }
            else:
                sp_loss_list = accumulator['sp_loss']
                accumulator['sp_loss'] = {}
                for i, layer_loss in enumerate(sp_loss_list):
                    accumulator['sp_loss']['layer'+str(i)] = layer_loss
                self.testing_history['perf'][self.round] = {'round': self.round, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                    'loss': accumulator['test_loss']/accumulator['test_len'],
                    'sp_loss': accumulator['sp_loss'],
                    'test_len': accumulator['test_len']
                }


            logging.info("FL Testing in epoch: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, sploss: {}, test loss: {:.4f}, test len: {}"
                    .format(self.round, self.global_virtual_clock, self.testing_history['perf'][self.round]['top_1'],
                    self.testing_history['perf'][self.round]['top_5'], self.testing_history['perf'][self.round]['sp_loss'],self.testing_history['perf'][self.round]['loss'],
                    self.testing_history['perf'][self.round]['test_len']))

            # Dump the testing result
            with open(os.path.join(logDir, 'testing_perf'), 'wb') as fout:
                pickle.dump(self.testing_history, fout)

            if len(self.loss_accumulator):
                self.log_writer.add_scalar('Test/round_to_loss', self.testing_history['perf'][self.round]['loss'], self.round)
                self.log_writer.add_scalar('Test/round_to_accuracy', self.testing_history['perf'][self.round]['top_1'], self.round)
                self.log_writer.add_scalar('FAR/time_to_test_loss (min)', self.testing_history['perf'][self.round]['loss'],
                                            self.global_virtual_clock/60.)
                self.log_writer.add_scalar('FAR/time_to_test_accuracy (min)', self.testing_history['perf'][self.round]['top_1'],
                                            self.global_virtual_clock/60.)
                self.log_writer.add_scalars('Test/sp_loss', self.testing_history['perf'][self.round]['sp_loss'], self.round)

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
        client_val_res = results['val_res']
        if clientId not in self.client_loss_accumulator.keys():
            self.client_loss_accumulator[clientId] = [client_val_res['val_loss']]
        else:
            self.client_loss_accumulator[clientId].append(client_val_res['val_loss'])

        self.client_manager.registerScore(results['clientId'], results['utility'],
            auxi=math.sqrt(results['moving_loss']),
            time_stamp=self.round,
            duration=self.virtual_client_clock[results['clientId']]['computation']+
                self.virtual_client_clock[results['clientId']]['communication']
        )

        device = self.device
        """
            [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
            H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
        """
        # Start to take the average of updates, and we do not keep updates to save memory
        # Importance of each update is 1/#_of_participants
        # importance = 1./self.tasks_round

        self.update_lock.acquire()
        # ================== Aggregate weights ======================

        self.model_in_update += 1
        if self.model_in_update == 1:
            for p in results['update_weight']:
                temp_list = results['update_weight'][p]
                if isinstance(results['update_weight'][p], list):
                    temp_list = np.asarray(temp_list, dtype=np.float32)
                self.model_weights[p].data = torch.from_numpy(temp_list).to(device=device)
        else:
            for p in results['update_weight']:
                temp_list = results['update_weight'][p]
                if isinstance(results['update_weight'][p], list):
                    temp_list = np.asarray(temp_list, dtype=np.float32)
                self.model_weights[p].data += torch.from_numpy(temp_list).to(device=device)

        if self.model_in_update == self.tasks_round:
            for p in self.model_weights:
                d_type = self.model_weights[p].data.dtype
                self.model_weights[p].data = (self.model_weights[p]/float(self.tasks_round)).to(dtype=d_type)
        self.update_lock.release()

if __name__ == "__main__":
    aggregator = Customized_Aggregator(args)
    aggregator.run()
