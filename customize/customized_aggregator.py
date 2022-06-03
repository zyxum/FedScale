import logging
import math
import torch
import numpy as np
from fedscale.core.aggregator import Aggregator
from customized_arg_parser import args
class Customized_Aggregator(Aggregator):

    def __init__(self, args):
        super().__init__(args)
        self.client_loss_accumulator = {}

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
        logging.info(f"finish client completion handler")

if __name__ == "__main__":
    aggregator = Customized_Aggregator(args)
    aggregator.run()
