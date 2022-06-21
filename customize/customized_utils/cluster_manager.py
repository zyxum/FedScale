from collections import defaultdict
import logging
import statistics

class Cluster_Manager(object):
    def __init__(self) -> None:
        self.clusters = defaultdict(list) # cluster_id: [client_ids, ...]
        self.val_loss = defaultdict(list) # client_id: [val_loss, ...]
        self.avg_val_loss = []

    def register_client(self, client_id):
        if client_id not in self.clusters[0]:
            self.clusters[0].append(client_id)
    
    def record_val_loss(self, val_loss: dict):
        avg_loss = 0.0
        for client_id in val_loss.keys():
            self.val_loss[client_id].append(val_loss[client_id])
            avg_loss += val_loss[client_id]
        avg_loss /= len(val_loss.keys())
        self.avg_val_loss.append(avg_loss)
        # at most 4 clusters
        if len(self.clusters) > 4:
            return False
        if len(self.avg_val_loss) > 3:
            if abs(self.avg_val_loss[-1] - self.avg_val_loss[-2]) <= 1:
                logging.info(f"average validation loss stagnates, it is time to split cluster, \n current value {abs(self.avg_val_loss[-1] - self.avg_val_loss[-2])}, target value {abs(self.avg_val_loss[-1] - self.avg_val_loss[0]) / 100000}")
                return True
            elif len(self.avg_val_loss) in [100, 200, 400, 600]:
                return True
            else:
                logging.info(f"average validation loss not stagnates, current value {abs(self.avg_val_loss[-1] - self.avg_val_loss[-2])}, target value {abs(self.avg_val_loss[-1] - self.avg_val_loss[0]) / 100000}")
        return False

    def split_cluster(self):
        picked_clients = {}
        picked_val_loss = []
        for client_id in self.clusters[len(self.clusters)-1]:
            picked_clients[client_id] = self.val_loss[client_id][-1]
            picked_val_loss.append(self.val_loss[client_id][-1])
        # get the median of picked clients
        median = statistics.median(picked_val_loss)
        # re-cluster
        new_cluster = len(self.clusters)
        for client_id in picked_clients.keys():
            if picked_clients[client_id] > median:
                self.clusters[new_cluster].append(client_id)
                self.clusters[new_cluster-1].pop(client_id)

    def query_cluster_id(self, client_id):
        for cluster_id in self.clusters.keys():
            if client_id in self.clusters[cluster_id]:
                return cluster_id
        logging.info(f'Cluster manager did not find client {client_id}')
        raise Exception
