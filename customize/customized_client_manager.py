from collections import defaultdict
import math
from random import Random
import pickle
import logging

import statistics

from fedscale.core.helper.client import Client

class customized_clientManager(object):

    def __init__(self, mode, args, sample_seed=233):
        self.Clients = {}
        self.clientOnHosts = {}
        self.mode = mode
        self.filter_less = args.filter_less
        self.filter_more = args.filter_more

        self.ucbSampler = None 

        if self.mode == 'oort': 
            import sys,os
            current = os.path.dirname(os.path.realpath(__file__))
            parent = os.path.dirname(current)
            sys.path.append(parent)
            from thirdparty.oort.oort import create_training_selector
            #sys.path.append(current) 
            self.ucbSampler =  create_training_selector(args=args)
        # self.feasibleClients = [[]] # feasible clients per cluster
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.count = 0
        self.feasible_samples = 0
        self.user_trace = None
        self.args = args

        # ==== for cluster management ====
        self.clusters = defaultdict(set)
        self.val_loss = defaultdict(list)
        self.train_loss = defaultdict(list)
        self.avg_val_loss = []
        self.round_duration = {}
        

        if args.device_avail_file is not None:
            with open(args.device_avail_file, 'rb') as fin:
                self.user_trace = pickle.load(fin)
            self.user_trace_keys = list(self.user_trace.keys())

    def registerClient(self, hostId, clientId, size, speed, duration=1):

        uniqueId = self.getUniqueId(hostId, clientId)
        user_trace = None if self.user_trace is None else self.user_trace[self.user_trace_keys[int(clientId)%len(self.user_trace)]]

        self.Clients[uniqueId] = Client(hostId, clientId, speed, user_trace)
        assert(str(clientId) in self.Clients.keys())

        # remove clients
        if size >= self.filter_less and size <= self.filter_more:
            # register clients in cluster
            assert(clientId not in self.clusters[0])
            self.clusters[0].add(clientId)
            # logging.info(f"get client {uniqueId}")
            # self.feasibleClients[0].append(clientId)
            self.feasible_samples += size

            if self.mode == "oort":
                feedbacks = {'reward':min(size, self.args.local_steps*self.args.batch_size),
                            'duration':duration,
                            }
                self.ucbSampler.register_client(clientId, feedbacks=feedbacks)
        else:
            del self.Clients[uniqueId]
            # logging.info(f"remove client {uniqueId}")

    def getAllClients(self):
        # return [clients for cluster in self.feasibleClients for clients in cluster]
        all_clients = []
        for clusterId in self.clusters.keys():
            all_clients += list(self.clusters[clusterId])
        return all_clients

    def getAllClientsLength(self):
        return len(self.getAllClients())

    def getClient(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)]

    def registerDuration(self, clientId, batch_size, upload_step, upload_size, download_size):
        if self.getUniqueId(0, clientId) in self.Clients:
            exe_cost = self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
                    batch_size=batch_size, upload_step=upload_step,
                    upload_size=upload_size, download_size=download_size
            )
            self.round_duration[clientId] = exe_cost['computation'] + exe_cost['communication']

    def getCompletionTime(self, clientId, batch_size, upload_step, upload_size, download_size):
        
        return self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
                batch_size=batch_size, upload_step=upload_step,
                upload_size=upload_size, download_size=download_size
            )

    def registerSpeed(self, hostId, clientId, speed):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId].speed = speed

    def registerScore(self, clientId, reward, auxi=1.0, time_stamp=0, duration=1., success=True):
        # currently, we only use distance as reward
        if self.mode == "oort":
            feedbacks = {
                'reward': reward,
                'duration': duration,
                'status': True,
                'time_stamp': time_stamp
            }

            self.ucbSampler.update_client_util(clientId, feedbacks=feedbacks)

    def registerClientScore(self, clientId, reward):
        self.Clients[self.getUniqueId(0, clientId)].registerReward(reward)

    def getScore(self, hostId, clientId):
        uniqueId = self.getUniqueId(hostId, clientId)
        return self.Clients[uniqueId].getScore()

    def getClientsInfo(self):
        clientInfo = {}
        for i, clientId in enumerate(self.Clients.keys()):
            client = self.Clients[clientId]
            clientInfo[client.clientId] = client.distance
        return clientInfo

    def nextClientIdToRun(self, hostId):
        init_id = hostId - 1
        lenPossible = self.getAllClientsLength()

        while True:
            clientId = str(self.getAllClientsLength())
            csize = self.Clients[clientId].size
            if csize >= self.filter_less and csize <= self.filter_more:
                return int(clientId)

            init_id = max(0, min(int(math.floor(self.rng.random() * lenPossible)), lenPossible - 1))

    def getUniqueId(self, hostId, clientId):
        return str(clientId)
        #return (str(hostId) + '_' + str(clientId))

    def clientSampler(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].size

    def clientOnHost(self, clientIds, hostId):
        self.clientOnHosts[hostId] = clientIds

    def getCurrentClientIds(self, hostId):
        return self.clientOnHosts[hostId]

    def getClientLenOnHost(self, hostId):
        return len(self.clientOnHosts[hostId])

    def getClientSize(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].size

    def getSampleRatio(self, clientId, hostId, even=False):
        totalSampleInTraining = 0.

        if not even:
            for key in self.clientOnHosts.keys():
                for client in self.clientOnHosts[key]:
                    uniqueId = self.getUniqueId(key, client)
                    totalSampleInTraining += self.Clients[uniqueId].size

            #1./len(self.clientOnHosts.keys())
            return float(self.Clients[self.getUniqueId(hostId, clientId)].size)/float(totalSampleInTraining)
        else:
            for key in self.clientOnHosts.keys():
                totalSampleInTraining += len(self.clientOnHosts[key])

            return 1./totalSampleInTraining

    def getFeasibleClients(self, cur_time, clusterId):
        if self.user_trace is None:
            clients_online = self.clusters[clusterId]
        else:
            clients_online = [clientId for clientId in self.clusters[clusterId] if self.Clients[self.getUniqueId(0, clientId)].isActive(cur_time)]

        logging.info(f"cluster: {clusterId}, Wall clock time: {round(cur_time)}, {len(clients_online)} clients online, " + \
                    f"{self.getAllClientsLength()-len(clients_online)} clients offline")

        return clients_online

    def isClientActive(self, clientId, cur_time):
        return self.Clients[self.getUniqueId(0, clientId)].isActive(cur_time)

    def resampleClients(self, numOfClients, clusterId, cur_time=0):
        self.count += 1

        clients_online = self.getFeasibleClients(cur_time, clusterId)

        if len(clients_online) <= numOfClients:
            return clients_online

        pickled_clients = None
        clients_online_set = set(clients_online)

        if self.mode == "oort" and self.count > 1:
            pickled_clients = self.ucbSampler.select_participant(numOfClients, feasible_clients=clients_online_set)
        else:
            self.rng.shuffle(clients_online)
            client_len = min(numOfClients, len(clients_online) -1)
            pickled_clients = clients_online[:client_len]

        return pickled_clients

    def getAllMetrics(self):
        if self.mode == "oort":
            return self.ucbSampler.getAllMetrics()
        return {}

    def getDataInfo(self):
        return {'total_feasible_clients': self.getAllClientsLength(), 'total_num_samples': self.feasible_samples}

    def getClientReward(self, clientId):
        return self.ucbSampler.get_client_reward(clientId)

    def get_median_reward(self):
        if self.mode == 'oort':
            return self.ucbSampler.get_median_reward()
        return 0.

    def register_loss(self, val_loss: dict, train_loss: dict):
        if len(val_loss) == 0:
            return
        avg_loss = 0.0
        for client_id in val_loss.keys():
            self.val_loss[client_id].append(val_loss[client_id])
            avg_loss += val_loss[client_id]
        for client_id in train_loss.keys():
            self.train_loss[client_id].append(train_loss[client_id])
        avg_loss /= len(val_loss.keys())
        self.avg_val_loss.append(avg_loss)
        # at most 4 clusters
        if len(self.clusters) > 4:
            return False
        if len(self.avg_val_loss) > 3:
            if abs(self.avg_val_loss[-1] - self.avg_val_loss[-2]) <= \
                abs(self.avg_val_loss[0] - self.avg_val_loss[1]) / 10:
                logging.info(f"average validation loss stagnates, it is time to split cluster, \n current value {abs(self.avg_val_loss[-1] - self.avg_val_loss[-2])}, target value {abs(self.avg_val_loss[0] - self.avg_val_loss[1]) / 10}")
                return True
            elif len(self.avg_val_loss) in [100, 200, 400, 600]:
                # enforce cluster
                logging.info("Warning: Enforce spilt cluster")
                return True
            else:
                logging.info(f"average validation loss not stagnates, current value {abs(self.avg_val_loss[-1] - self.avg_val_loss[-2])}, target value {abs(self.avg_val_loss[0] - self.avg_val_loss[1]) / 10}")
        return False
    
    def cal_utility(self, clientId):
        if self.val_loss[clientId] == [] or clientId not in self.val_loss.keys():
            return -1
        else:
            return (self.val_loss[clientId][-1] + self.train_loss[clientId][-1])\
                * self.round_duration[clientId]
    
    def split_cluster(self):
        # get the clients' id and scores of the newest cluster
        picked_clients = {}
        picked_scores = []
        for client_id in self.clusters[len(self.clusters)-1]:
            score = self.cal_utility(client_id)
            if score != -1:
                picked_clients[client_id] = score
                picked_scores.append(score)
        # get the median of picked clients
        median = statistics.median(picked_scores)
        # re-cluster
        new_cluster = len(self.clusters)
        for client_id in picked_clients.keys():
            if picked_clients[client_id] > median:
                self.clusters[new_cluster].add(client_id)
                self.clusters[new_cluster-1].remove(client_id)
        logging.info(f"re-splitted cluster: {self.clusters[new_cluster]}")
    
    def query_cluster_id(self, client_id: str):
        for cluster_id in self.clusters.keys():
            if str(client_id) in self.clusters[cluster_id] or\
                int(client_id) in self.clusters[cluster_id]:
                return cluster_id
        logging.info(f'client manager did not find client {client_id}')
        logging.info(self.clusters)
        raise Exception

    def get_cluster_worker(self, clusterId, total_worker):
        assert(clusterId in self.clusters.keys())
        # consider fairness among clusters
        return int(total_worker * len(self.clusters[clusterId]) / self.getAllClientsLength())


