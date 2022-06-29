from collections import defaultdict
from fedscale.core import events
import threading
from fedscale.core.resource_manager import ResourceManager

class Customized_ResourceManager(ResourceManager):

    def __init__(self, experiment_mode):
        self.client_run_queue = defaultdict(list)
        self.client_run_queue_idx = [0]
        self.experiment_mode = experiment_mode
        self.update_lock = threading.Lock()

    def register_tasks(self, clientToRun, clusterId):
        self.client_run_queue[clusterId] = clientToRun.copy()
        self.client_run_queue_idx[clusterId] = 0

    def remove_client_task(self, client_id, clusterId):
        assert(client_id in self.client_run_queue[clusterId],
            f"client task {client_id} is not in task queue")
        pass

    def has_next_task(self, clusterId: int, client_id=None):
        exist_next_task = False
        if self.experiment_mode == events.SIMULATION_MODE:
            exist_next_task = self.client_run_queue_idx[clusterId] < len(self.client_run_queue[clusterId])
        else:
            exist_next_task = client_id in self.client_run_queue[clusterId]
        return exist_next_task
    
    def get_next_task(self, clusterId: int, client_id=None):
        next_task_id = None
        self.update_lock.acquire()
        if self.experiment_mode == events.SIMULATION_MODE:
            if self.has_next_task(clusterId, client_id):
                next_task_id = self.client_run_queue[clusterId][self.client_run_queue_idx[clusterId]]
                self.client_run_queue_idx[clusterId] += 1
        else:
            if client_id in self.client_run_queue[clusterId]:
                next_task_id = client_id
                self.client_run_queue[clusterId].remove(next_task_id)

        self.update_lock.release()
        return next_task_id
        