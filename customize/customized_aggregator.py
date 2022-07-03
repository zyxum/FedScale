import collections, logging, math, os, sys, pickle, threading, time, random
from copy import deepcopy
import numpy as np
import grpc
from concurrent import futures
import torch
from torch.utils.tensorboard import SummaryWriter


from fedscale.core import events
from fedscale.core.fl_aggregator_libs import logDir
from fedscale.core.aggregator import Aggregator
from fedscale.core.fllibs import init_model
from fedscale.core.optimizer import ServerOptimizer
import fedscale.core.job_api_pb2_grpc as job_api_pb2_grpc
from fedscale.core import job_api_pb2


from customized_arg_parser import args
from customized_resource_manager import Customized_ResourceManager
from customized_utils.architecture_manager import Architecture_Manager
from customized_client_manager import customized_clientManager

MAX_MESSAGE_LENGTH = 1*1024*1024*1024 # 1GB

class Customized_Aggregator(Aggregator):

    def __init__(self, args):
        logging.info(f"Job args {args}")

        self.args = args
        self.experiment_mode = args.experiment_mode
        self.device = args.cuda_device if args.use_cuda else torch.device('cpu')

        # ======== env information ========
        self.this_rank = 0
        self.resource_manager = Customized_ResourceManager(self.experiment_mode)
        self.client_manager = self.init_client_manager(args=args)

        # ======== model and data ========
        # self.models = None
        # self.model_in_update = 0
        self.update_lock = threading.Lock()
        # self.model_weights = collections.OrderedDict() # all weights including bias/#_batch_tracked (e.g., state_dict)
        self.last_gradient_weights = [[]] # only gradient variables
        self.model_state_dict = None
        # NOTE: if <param_name, param_tensor> (e.g., model.parameters() in PyTorch), then False
        # True, if <param_name, list_param_tensors> (e.g., layer.get_weights() in Tensorflow)
        self.using_group_params = self.args.engine == events.TENSORFLOW 

        # ======== channels ========
        self.connection_timeout = self.args.connection_timeout
        self.executors = None
        self.grpc_server = None

        # ======== Event Queue =======
        self.individual_client_events = {}    # Unicast
        self.sever_events_queue = collections.deque()
        self.broadcast_events_queue = collections.deque() # Broadcast

        # ======== runtime information ========
        self.tasks_round = 0
        self.num_of_clients = 0

        # NOTE: sampled_participants = sampled_executors in deployment,
        # because every participant is an executor. However, in simulation mode,
        # executors is the physical machines (VMs), thus:
        # |sampled_executors| << |sampled_participants| as an VM may run multiple participants
        self.sampled_participants = [[]]
        self.sampled_executors = []

        self.round_stragglers = [[]]
        self.model_update_size = 0.

        self.collate_fn = None
        self.task = args.task
        # self.round = 0

        self.start_run_time = time.time()
        self.client_conf = {}

        self.stats_util_accumulator = [[]]
        self.loss_accumulator = [[]]
        # self.client_training_results = []

        # number of registered executors
        self.registered_executor_info = set()
        self.test_result_accumulator = [[]]
        self.testing_history = [{'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
                        'gradient_policy': args.gradient_policy, 'task': args.task, 'perf': collections.OrderedDict()}]

        self.log_writer = SummaryWriter(log_dir=logDir)

        # ======== Task specific ============
        self.init_task_context()

        # ======== Cluster specific =========
        self.models = [None]
        self.model_weights = [collections.OrderedDict()]
        self.cluster_virtual_clocks = [0.]
        self.cluster_round_duration = [0.]
        self.client_val_loss_in_update = [{}]
        self.client_train_loss_in_update = [{}]
        self.model_in_update = [0]
        self.round_duration = [0.]
        self.round = [0]
        self.cluster_worker = [args.total_worker]
        self.tasks_cluster = [0]
        self.need_update = False
        self.flatten_client_duration = {}
        self.virtual_client_clock = [{}]
        # self.cluster_manager = Cluster_Manager()


    def setup_env(self):
        self.setup_seed(seed=1)
        self.optimizer = ServerOptimizer(self.args.gradient_policy, self.args, self.device)


    def setup_seed(self, seed=1):
        """Set global random seed for better reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def init_control_communication(self):
        # Create communication channel between aggregator and worker
        # This channel serves control messages
        logging.info(f"Initiating control plane communication ...")
        if self.experiment_mode == events.SIMULATION_MODE:
            num_of_executors = 0
            for ip_numgpu in self.args.executor_configs.split("="):
                ip, numgpu = ip_numgpu.split(':')
                for numexe in numgpu.strip()[1:-1].split(','):
                    for _ in range(int(numexe.strip())):
                        num_of_executors += 1
            self.executors = list(range(num_of_executors))
        else:
            self.executors = list(range(self.args.total_worker))

        # initiate a server process
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ],
        )
        job_api_pb2_grpc.add_JobServiceServicer_to_server(self, self.grpc_server)
        port = '[::]:{}'.format(self.args.ps_port)

        logging.info(f'%%%%%%%%%% Opening aggregator sever using port {port} %%%%%%%%%%')

        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()


    def init_data_communication(self):
        """For jumbo traffics (e.g., training results).
        """
        pass


    def init_model(self):
        assert self.args.engine == events.PYTORCH, "Please define model for non-PyTorch models"
        self.models = [init_model()]
        self.model_weights = [self.models[0].state_dict()]

    
    def init_client_manager(self, args):
        """
            Currently we implement two client managers:
            1. Random client sampler
                - it selects participants randomly in each round
                - [Ref]: https://arxiv.org/abs/1902.01046
            2. Oort sampler
                - Oort prioritizes the use of those clients who have both data that offers the greatest utility
                  in improving model accuracy and the capability to run training quickly.
                - [Ref]: https://www.usenix.org/conference/osdi21/presentation/lai
        """

        # sample_mode: random or oort
        client_manager = customized_clientManager(args.sample_mode, args=args)

        return client_manager


    def load_client_profile(self, file_path):
        """For Simulation Mode: load client profiles/traces"""
        global_client_profile = {}
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fin:
                # {clientId: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        return global_client_profile
    

    def client_register_handler(self, executorId, info):
        """Triggered once receive new executor registration"""

        logging.info(f"Loading {len(info['size'])} client traces ...")
        for _size in info['size']:
            # since the worker rankId starts from 1, we also configure the initial dataId as 1
            mapped_id = (self.num_of_clients+1)%len(self.client_profiles) if len(self.client_profiles) > 0 else 1
            systemProfile = self.client_profiles.get(mapped_id, {'computation': 1.0, 'communication':1.0})

            clientId = (self.num_of_clients+1) if self.experiment_mode == events.SIMULATION_MODE else executorId
            self.client_manager.registerClient(executorId, clientId, size=_size, speed=systemProfile)
            self.client_manager.registerDuration(clientId, batch_size=self.args.batch_size,
                upload_step=self.args.local_steps, upload_size=self.model_update_size, download_size=self.model_update_size)
            self.num_of_clients += 1

        logging.info("Info of all feasible clients {}".format(self.client_manager.getDataInfo()))


    def executor_info_handler(self, executorId, info):

        self.registered_executor_info.add(executorId)
        logging.info(f"Received executor {executorId} information, {len(self.registered_executor_info)}/{len(self.executors)}")

        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout
        if self.experiment_mode == events.SIMULATION_MODE:

            if len(self.registered_executor_info) == len(self.executors):
                self.client_register_handler(executorId, info)
                # start to sample clients
                logging.info(f"init by running round completion handler")
                self.round_completion_handler(0)
        else:
            # In real deployments, we need to register for each client
            self.client_register_handler(executorId, info)
            if len(self.registered_executor_info) == len(self.executors):
                self.round_completion_handler(0)

    
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
                cluster_id = self.client_manager.query_cluster_id(client_to_run)
                exe_cost = self.client_manager.getCompletionTime(client_to_run,
                                        batch_size=client_cfg.batch_size, upload_step=client_cfg.local_steps,
                                        upload_size=self.model_update_size, download_size=self.model_update_size)

                roundDuration = exe_cost['computation'] + exe_cost['communication']
                # if the client is not active by the time of collection, we consider it is lost in this round
                if self.client_manager.isClientActive(client_to_run, roundDuration + self.cluster_virtual_clocks[cluster_id]):
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

            logging.info(f"after tictak: clients_to_run: {client_to_run}, dummy_clients: {dummy_clients}")

            return (clients_to_run, dummy_clients, 
                    completed_client_clock, round_duration, 
                    completionTimes[:num_clients_to_collect])
        else:
            completed_client_clock = {
                client:{'computation': 1, 'communication':1} for client in sampled_clients}
            completionTimes = [1 for c in sampled_clients]
            return (sampled_clients, sampled_clients, completed_client_clock, 
                1, completionTimes)


    def run(self):
        self.setup_env()
        self.init_control_communication()
        self.init_data_communication()

        self.init_model()
        self.save_last_param(0)
        self.model_update_size = sys.getsizeof(pickle.dumps(self.models[0]))/1024.0*8. # kbits
        self.client_profiles = self.load_client_profile(file_path=self.args.device_conf_file)

        dummy_input = torch.randn(10, 3, 32, 32, device=self.device)
        self.archi_manager = Architecture_Manager(dummy_input, 'customize_server.onnx')
        self.archi_manager.parse_model(self.models[-1])

        self.event_monitor()

    
    def select_participants(self, select_num_participants, overcommitment=1.3, cluster_id=0):
        return sorted(self.client_manager.resampleClients(
            int(select_num_participants*overcommitment), 
            cluster_id,
            cur_time=self.cluster_virtual_clocks[cluster_id]),
        )
    
    
    def client_completion_handler(self, results):
        """We may need to keep all updates from clients, if so, we need to append results to the cache"""
        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility
        #       'val_res': val_res}

        # not support q-fedavg for now

        # Feed metrics to client sampler
        clientId = results['clientId']
        clusterId = self.client_manager.query_cluster_id(clientId)
        client_val_res = results['val_res']
        self.stats_util_accumulator[clusterId].append(results['utility'])
        self.loss_accumulator[clusterId].append(results['moving_loss'])
        self.client_val_loss_in_update[clusterId][clientId] = client_val_res['val_loss']
        self.client_train_loss_in_update[clusterId][clientId] = results['moving_loss']

        # this is only for oort
        self.client_manager.registerScore(results['clientId'], results['utility'],
            auxi=math.sqrt(results['moving_loss']),
            time_stamp=self.round[clusterId],
            duration=self.virtual_client_clock[clusterId][results['clientId']]['computation']+
                self.virtual_client_clock[clusterId][results['clientId']]['communication']
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

    def save_last_param(self, clusterId):
        if self.args.engine == events.TENSORFLOW:
            self.last_gradient_weights[clusterId] = [layer.get_weights() for layer in self.model.layers]
        else:
            if clusterId == len(self.last_gradient_weights):
                self.last_gradient_weights.append([p.data.clone() for p in self.models[clusterId].parameters()])
            self.last_gradient_weights[clusterId] = [p.data.clone() for p in self.models[clusterId].parameters()]


    def round_weight_handler(self, last_model, clusterId):
        """Update model when the round completes"""
        if self.round[clusterId] > 1:
            if self.args.engine == events.TENSORFLOW:
                for layer in self.model.layers:
                    layer.set_weights([p.cpu().detach().numpy() for p in self.model_weights[layer.name]])
                # TODO: support update round gradient
            else:
                self.models[clusterId].load_state_dict(self.model_weights[clusterId])
                # temporarily disable server optimizer
                # current_grad_weights = [param.data.clone() for param in self.model.parameters()]
                # self.optimizer.update_round_gradient(last_model, current_grad_weights, self.model)


    def round_completion_handler(self, clusterId):
        self.cluster_virtual_clocks[clusterId] += self.round_duration[clusterId]
        self.round[clusterId] += 1

        if self.round[clusterId] % self.args.decay_round == 0:
            self.args.learning_rate = max(self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

        # handle the global update w/ current and last
        self.round_weight_handler(self.last_gradient_weights, clusterId)

        avgUtilLastround = sum(self.stats_util_accumulator[clusterId])/max(1, len(self.stats_util_accumulator[clusterId]))
        # assign avg reward to explored, but not ran workers
        for clientId in self.round_stragglers[clusterId]:
            self.client_manager.registerScore(clientId, avgUtilLastround,
                    time_stamp=self.round[clusterId],
                    duration=self.virtual_client_clock[clusterId][clientId]['computation']+self.virtual_client_clock[clusterId][clientId]['communication'],
                    success=False)
        
        avg_loss = sum(self.loss_accumulator[clusterId])/max(1, len(self.loss_accumulator[clusterId]))
        logging.info(f"Cluster: {clusterId}, Wall clock: {round(self.cluster_virtual_clocks[clusterId])} s, round: {self.round[clusterId]}, Planned participants: " + \
            f"{len(self.sampled_participants[clusterId])}, Succeed participants: {len(self.stats_util_accumulator[clusterId])}, Training loss: {avg_loss}")
        
        # dump round completion information to tensorboard
        if len(self.loss_accumulator[clusterId]):
            self.log_train_result(avg_loss, clusterId)

        # calculate number of workers in cluster
        self.cluster_worker[clusterId] = self.client_manager.get_cluster_worker(clusterId, self.args.total_worker)
        logging.info(f"Cluster: {clusterId}, ready to sample {self.cluster_worker[clusterId]} participants")
        # update select participants
        self.sampled_participants[clusterId] = self.select_participants(
                        select_num_participants=self.cluster_worker[clusterId], overcommitment=self.args.overcommitment, cluster_id=clusterId)
        logging.info(f"Cluster: {clusterId}. Sampled participants to run: {self.sampled_participants[clusterId]}")
        (clientsToRun, round_stragglers, virtual_client_clock, round_duration, flatten_client_duration) = self.tictak_client_tasks(
                        self.sampled_participants[clusterId], self.cluster_worker[clusterId])
        logging.info(f"Cluster: {clusterId}. Selected participants to run: {clientsToRun}")


        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun, clusterId)
        self.tasks_cluster[clusterId] = len(clientsToRun)

        # Update executors and participants
        if self.experiment_mode == events.SIMULATION_MODE:
            self.sampled_executors = list(self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id) for c_id in self.sampled_participants[clusterId]]

        # record validate loss to cluster manager
        if len(self.client_val_loss_in_update) != 0 and not self.need_update\
            and clusterId == len(self.client_manager.clusters) - 1:
            self.need_update = self.client_manager.register_loss(
                self.client_val_loss_in_update[clusterId],
                self.client_train_loss_in_update[clusterId])
            if self.need_update:
                # prepare to split cluster
                logging.info(f"splitting cluster")
                self.client_manager.split_cluster()
                self.stats_util_accumulator.append([])
                self.loss_accumulator.append([])
                self.test_result_accumulator.append([])
                self.testing_history.append(deepcopy(self.testing_history[-1]))
                self.cluster_virtual_clocks.append(deepcopy(self.cluster_virtual_clocks[-1]))
                self.cluster_round_duration.append(0.)
                self.client_val_loss_in_update.append({})
                self.client_train_loss_in_update.append({})
                self.model_in_update.append(0)
                self.round_duration.append(0.)
                self.round.append(deepcopy(self.round[-1]))
                self.cluster_worker.append(0)
                self.tasks_cluster.append(0)
                self.round_stragglers.append([])
                self.sampled_participants.append([])
                self.virtual_client_clock.append(None)


        self.save_last_param(clusterId)

        self.round_stragglers[clusterId] = round_stragglers
        self.virtual_client_clock[clusterId] = virtual_client_clock
        self.flatten_client_duration[clusterId] = np.array(flatten_client_duration)
        self.round_duration[clusterId] = round_duration
        self.model_in_update[clusterId] = 0
        self.test_result_accumulator[clusterId] = []
        self.stats_util_accumulator[clusterId] = []
        # self.client_training_results = []
        self.client_val_loss_in_update[clusterId] = {}
        self.client_train_loss_in_update[clusterId] = {}

        if self.round[clusterId] >= self.args.rounds:
            self.broadcast_aggregator_events(events.SHUT_DOWN)
        elif self.round[clusterId] % self.args.eval_interval == 0:
            self.broadcast_aggregator_events(events.UPDATE_MODEL, clusterId=clusterId)
            self.broadcast_aggregator_events(events.MODEL_TEST, clusterId=clusterId)
        elif self.need_update and clusterId == len(self.client_manager.clusters) - 1:
            self.broadcast_aggregator_events(events.UPDATE_MODEL, clusterId=clusterId)
            self.broadcast_aggregator_events(events.MODEL_TEST, clusterId=clusterId)
        else:
            self.broadcast_aggregator_events(events.UPDATE_MODEL, clusterId=clusterId)
            self.broadcast_aggregator_events(events.START_ROUND, clusterId=clusterId)
        
        logging.info(f"finished broadcast events")


    def log_train_result(self, avg_loss, clusterId):
        """Result will be post on TensorBoard"""
        self.log_writer.add_scalar('Train/round_to_loss_' + str(clusterId), avg_loss, self.round[clusterId])
        self.log_writer.add_scalar('FAR/time_to_train_loss_' + str(clusterId) + ' (min)', avg_loss, self.cluster_virtual_clocks[clusterId]/60.)
        self.log_writer.add_scalar('FAR/round_duration_' + str(clusterId) + ' (min)', self.round_duration[clusterId]/60., self.round[clusterId])
        self.log_writer.add_histogram('FAR/client_duration_' + str(clusterId) + ' (min)', self.flatten_client_duration[clusterId], self.round[clusterId])


    def testing_completion_handler(self, client_id, results):
        """Each executor will handle a subset of testing dataset"""

        clusterId = results['clusterId']
        results = results['results']


        # List append is thread-safe
        self.test_result_accumulator[clusterId].append(results)

        # Have collected all testing results
        if len(self.test_result_accumulator[clusterId]) == len(self.executors):
            accumulator = self.test_result_accumulator[clusterId][0]
            for i in range(1, len(self.test_result_accumulator[clusterId])):
                if self.args.task == "detection":
                    for key in accumulator:
                        if key == "boxes":
                            for j in range(self.imdb.num_classes):
                                accumulator[key][j] = accumulator[key][j] + self.test_result_accumulator[clusterId][i][key][j]
                        else:
                            accumulator[key] += self.test_result_accumulator[clusterId][i][key]
                else:
                    for key in accumulator:
                        if isinstance(accumulator[key], dict):
                            for subkey in accumulator[key].keys():
                                accumulator[key][subkey] += self.test_result_accumulator[clusterId][i][key][subkey]
                        else:
                            accumulator[key] += self.test_result_accumulator[clusterId][i][key]
            if self.args.task == "detection":
                self.testing_history[clusterId]['perf'][self.round[clusterId]] = {'round': self.round[clusterId], 'clock': self.cluster_virtual_clocks[clusterId],
                    'top_1': round(accumulator['top_1']*100.0/len(self.test_result_accumulator[clusterId]), 4),
                    'top_5': round(accumulator['top_5']*100.0/len(self.test_result_accumulator[clusterId]), 4),
                    'loss': accumulator['test_loss'],
                    'test_len': accumulator['test_len']
                }
            else:
                self.testing_history[clusterId]['perf'][self.round[clusterId]] = {'round': self.round[clusterId], 'clock': self.cluster_virtual_clocks[clusterId],
                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                    'loss': accumulator['test_loss']/accumulator['test_len'],
                    'sp_loss': accumulator['sp_loss'],
                    'test_len': accumulator['test_len']
                }


            logging.info("Cluster: {}. FL Testing in epoch: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, sploss: {}, test loss: {:.4f}, test len: {}"
                    .format(clusterId, self.round[clusterId], self.cluster_virtual_clocks[clusterId], self.testing_history[clusterId]['perf'][self.round[clusterId]]['top_1'],
                    self.testing_history[clusterId]['perf'][self.round[clusterId]]['top_5'], self.testing_history[clusterId]['perf'][self.round[clusterId]]['sp_loss'],self.testing_history[clusterId]['perf'][self.round[clusterId]]['loss'],
                    self.testing_history[clusterId]['perf'][self.round[clusterId]]['test_len']))

            # Dump the testing result
            with open(os.path.join(logDir, 'testing_perf_' + str(clusterId)), 'wb') as fout:
                pickle.dump(self.testing_history[clusterId], fout)

            if len(self.loss_accumulator[clusterId]):
                self.log_writer.add_scalar('Test/round_to_loss_' + str(clusterId), self.testing_history[clusterId]['perf'][self.round[clusterId]]['loss'], self.round[clusterId])
                self.log_writer.add_scalar('Test/round_to_accuracy_' + str(clusterId), self.testing_history[clusterId]['perf'][self.round[clusterId]]['top_1'], self.round[clusterId])
                self.log_writer.add_scalar('FAR/time_to_test_loss_' + str(clusterId) + ' (min)', self.testing_history[clusterId]['perf'][self.round[clusterId]]['loss'],
                                            self.cluster_virtual_clocks[clusterId]/60.)
                self.log_writer.add_scalar('FAR/time_to_test_accuracy_' + str(clusterId) + ' (min)', self.testing_history[clusterId]['perf'][self.round[clusterId]]['top_1'],
                                            self.cluster_virtual_clocks[clusterId]/60.)
                self.log_writer.add_scalars('Test/sp_loss_' + str(clusterId), self.testing_history[clusterId]['perf'][self.round[clusterId]]['sp_loss'], self.round[clusterId])

            # widen layer on demand
            if self.need_update and len(self.testing_history[clusterId]['perf'][self.round[clusterId]]['sp_loss']) > 0:
                model = self.archi_manager.widen(self.testing_history[clusterId]['perf'][self.round[clusterId]]['sp_loss'], self.models[-1])
                self.models.append(model)
                self.model_weights.append(model.state_dict())
                self.need_update = False
                self.round_completion_handler(len(self.models) - 1)


            self.broadcast_events_queue.append((events.START_ROUND, clusterId))

    def broadcast_aggregator_events(self, event, clusterId=-1):
        """Issue tasks (events) to aggregator worker processes"""
        self.broadcast_events_queue.append((event, clusterId))


    def dispatch_client_events(self, event, clusterId=-1, clients=None):
        """Issue tasks (events) to clients"""
        if clients is None:
            clients = self.sampled_executors

        for client_id in clients:
            self.individual_client_events[client_id].append((event, clusterId))

    
    def get_client_conf(self, clusterId):
        """Training configurations that will be applied on clients"""

        conf = {
            'learning_rate': self.args.learning_rate,
            'model': None # none indicates we are using the global model
        }
        return conf


    def create_client_task(self, executorId, clusterId):
        """Issue a new client training task to the executor"""

        next_clientId = self.resource_manager.get_next_task(clusterId, executorId)
        train_config = None
        # NOTE: model = None then the executor will load the global model broadcasted in UPDATE_MODEL
        model = None
        if next_clientId != None:
            config = self.get_client_conf(clusterId)
            train_config = {'client_id': next_clientId, 'task_config': config, 'model_id': clusterId}
        return train_config, model


    def get_test_config(self, clusterId):
        """FL model testing on clients"""

        return {'model_id': clusterId}

    def get_global_model(self, clusterId=-1):
        """Get global model that would be used by all FL clients (in default FL)"""
        return self.models[clusterId]


    def CLIENT_REGISTER(self, request, context):
        """FL Client register to the aggregator"""

        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id uses the same executor_id (VMs) in simulations
        executor_id = request.executor_id
        executor_info = self.deserialize_response(request.executor_info)
        if executor_id not in self.individual_client_events:
            #logging.info(f"Detect new client: {executor_id}, executor info: {executor_info}")
            self.individual_client_events[executor_id] = collections.deque()
        else:
            logging.info(f"Previous client: {executor_id} resumes connecting")

        # We can customize whether to admit the clients here
        self.executor_info_handler(executor_id, executor_info)
        dummy_data = self.serialize_response(events.DUMMY_RESPONSE)

        return job_api_pb2.ServerResponse(event=events.DUMMY_EVENT,
                meta=dummy_data, data=dummy_data)


    def CLIENT_PING(self, request, context):
        """Handle client requests"""

        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id may use the same executor_id (VMs) in simulations
        executor_id, client_id = request.executor_id, request.client_id
        response_data = response_msg = events.DUMMY_RESPONSE

        if len(self.individual_client_events[executor_id]) == 0:
            # send dummy response
            current_event = events.DUMMY_EVENT
            response_data = response_msg = events.DUMMY_RESPONSE
        else:
            current_event, current_clusterId = self.individual_client_events[executor_id].popleft()
            assert(type(current_clusterId) == int)
            if current_event == events.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(client_id, current_clusterId)
                if response_msg is None:
                    current_event = events.DUMMY_EVENT
                    if self.experiment_mode != events.SIMULATION_MODE:
                        self.individual_client_events[executor_id].appendleft((events.CLIENT_TRAIN, current_clusterId))
            elif current_event == events.MODEL_TEST:
                response_msg = self.get_test_config(current_clusterId)
            elif current_event == events.UPDATE_MODEL:
                response_data = self.get_global_model(current_clusterId)
                response_msg = current_clusterId
            elif current_event == events.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)

        if current_event != events.DUMMY_EVENT:
            logging.info(f"Issue EVENT ({current_event}) to CLIENT ({executor_id}) at CLUSTER ({current_clusterId})")
        response_msg, response_data = self.serialize_response(response_msg), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        return job_api_pb2.ServerResponse(event=current_event,
                meta=response_msg, data=response_data)

    
    def CLIENT_EXECUTE_COMPLETION(self, request, context):
        """FL clients complete the execution task."""

        executor_id, client_id, event = request.executor_id, request.client_id, request.event
        execution_status, execution_msg = request.status, request.msg
        meta_result, data_result = request.meta_result, request.data_result

        if event == events.CLIENT_TRAIN:
            clusterId = self.client_manager.query_cluster_id(client_id)
            # Training results may be uploaded in CLIENT_EXECUTE_RESULT request later,
            # so we need to specify whether to ask client to do so (in case of straggler/timeout in real FL).
            if execution_status is False:
                logging.error(f"Executor {executor_id} fails to run client {client_id}, due to {execution_msg}")
            logging.info(f"query cluster {clusterId} for next task")
            if self.resource_manager.has_next_task(clusterId, executor_id):
                # NOTE: we do not pop the train immediately in simulation mode,
                # since the executor may run multiple clients
                self.individual_client_events[executor_id].appendleft((events.CLIENT_TRAIN, clusterId))

        elif event in (events.MODEL_TEST, events.UPLOAD_MODEL):
            self.add_event_handler(client_id, event, meta_result, data_result)
        else:
            logging.error(f"Received undefined event {event} from client {client_id}")
        return self.CLIENT_PING(request, context)


    def event_monitor(self):
        logging.info("Start monitoring events ...")

        while True:
            # Broadcast events to clients
            if len(self.broadcast_events_queue) > 0:
                current_event, current_cluster = self.broadcast_events_queue.popleft()

                if current_event in (events.UPDATE_MODEL, events.MODEL_TEST):
                    self.dispatch_client_events(current_event, clusterId=current_cluster)

                elif current_event == events.START_ROUND:
                    self.dispatch_client_events(events.CLIENT_TRAIN, clusterId=current_cluster)

                elif current_event == events.SHUT_DOWN:
                    self.dispatch_client_events(events.SHUT_DOWN)
                    break

            # Handle events queued on the aggregator
            elif len(self.sever_events_queue) > 0:
                client_id, current_event, meta, data = self.sever_events_queue.popleft()
                if current_event == events.UPLOAD_MODEL:
                    clusterId = self.client_manager.query_cluster_id(client_id)
                    self.client_completion_handler(self.deserialize_response(data))
                    if len(self.stats_util_accumulator[clusterId]) == self.tasks_cluster[clusterId]:
                            self.round_completion_handler(clusterId)

                elif current_event == events.MODEL_TEST:
                    self.testing_completion_handler(client_id, self.deserialize_response(data))
                    # here client_id is useless

                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)



if __name__ == "__main__":
    aggregator = Customized_Aggregator(args)
    aggregator.run()
