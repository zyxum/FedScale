import logging, gc, time, collections, os, random, pickle
import torch
import numpy as np
from argparse import Namespace


from fedscale.core import events
from fedscale.core.executor import Executor
from fedscale.core.fllibs import init_model
from fedscale.core.fl_client_libs import logDir
from fedscale.core.communication.channel_context import ClientConnections
import fedscale.core.job_api_pb2 as job_api_pb2

from customized_utils.architecture_manager import Architecture_Manager
from customized_arg_parser import args
from customized_fllib import init_dataset
from customized_utils.customized_utils_models import validate_model, test_model
from customized_client import Customized_Client
from customized_utils.customized_divide_data import Customized_DataPartitioner, select_dataset
from config import cfg

class Customized_Executor(Executor):
    def __init__(self, args):
        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device('cpu')
        self.num_executors = args.num_executors
        # ======== env information ========
        self.this_rank = args.this_rank
        self.executor_id = str(self.this_rank)

        # ======== model and data ========
        self.model = self.training_sets = self.test_dataset = None
        self.temp_model_path = os.path.join(logDir, 'model_'+str(args.this_rank)+'.pth.tar')

        # ======== channels ========
        self.aggregator_communicator = ClientConnections(args.ps_ip, args.ps_port)

        # ======== runtime information ========
        self.collate_fn = None
        self.task = args.task
        self.round = [0]
        self.start_run_time = time.time()
        self.received_stop_request = False
        self.event_queue = collections.deque()

        self.klayers_outputs = [[]]
        self.sploss_gap = cfg['sploss_gap']


    def setup_env(self):
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")
        self.setup_seed(seed=1)


    def setup_communication(self):
        self.init_control_communication()
        self.init_data_communication()


    def setup_seed(self, seed=1):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages."""
        self.aggregator_communicator.connect_to_server()


    def init_data_communication(self):
        """In charge of jumbo data traffics (e.g., fetch training result)
        """
        pass

    
    def init_model(self):
        """Return the model architecture used in training"""
        assert self.args.engine == events.PYTORCH, "Please override this function to define non-PyTorch models"
        model = init_model()
        model = model.to(device=self.device)
        return model
    
    
    def init_data(self):
        """Return the training, testing and validation dataset"""
        train_dataset, test_dataset, val_dataset = init_dataset()
        # disable rl
        logging.info("Data partitioner starts ...")

        training_sets = Customized_DataPartitioner(data=train_dataset, args = self.args, numOfClass=self.args.num_class)
        self.unique_clientId = training_sets.partition_data_helper(num_clients=self.args.total_worker, data_map_file=self.args.data_map_file)

        testing_sets = Customized_DataPartitioner(data=test_dataset, args = self.args, numOfClass=self.args.num_class, isTest=True)
        testing_sets.partition_data_helper(num_clients=self.num_executors)

        val_sets = Customized_DataPartitioner(data=val_dataset, args = self.args, numOfClass=self.args.num_class)
        val_sets.partition_data_helper(num_clients=self.args.total_worker, data_map_file=self.args.val_data_map_file, unique_clientId=self.unique_clientId)

        logging.info("Data partitioner competes ...")

        # not support nlp or voice tasks

        return training_sets, testing_sets, val_sets


    def run(self):
        self.setup_env()
        self.models = [self.init_model()]
        self.models = [model.to(device=self.device) for model in self.models]
        self.training_sets, self.testing_sets, self.val_sets = self.init_data()
        self.setup_communication()

        dummy_input = torch.randn(10, 3, 32, 32, device=self.device)
        self.archi_manager = Architecture_Manager(dummy_input, 'customize_clients.onnx')
        self.archi_manager.parse_model(self.models[0])
        self.event_monitor()
    
    
    def dispatch_worker_events(self, request):
        """Add new events to worker queues"""
        self.event_queue.append(request)


    def UpdateModel(self, config, model_id):
        """Receive the broadcasted global model for current round"""
        self.update_model_handler(config, model_id)


    def Train(self, config):
        """Integrate validation into training"""
        client_id, train_config = config['client_id'], config['task_config']
        train_config['model_id'] = config['model_id']
        model = None
        if 'model' in train_config and train_config['model'] is not None:
            model = train_config['model']
        
        client_conf = self.override_conf(train_config)
        train_res, model = self.training_handler(clientId=client_id, conf=client_conf, model=model)

        # validation
        assert(model is not None)
        # client_model.load_state_dict(train_res['update_weight'])
        val_res = self.validation_handler(client_id, model)

        train_res['val_res'] = val_res

        # Report execution completion meta information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(job_api_pb2.CompleteRequest(
            client_id = str(client_id), executor_id = self.executor_id,
            event = events.CLIENT_TRAIN, status = True, msg = None,
            meta_result = None, data_result = None
        ))
        self.dispatch_worker_events(response)

        return client_id, train_res


    def Test(self, config):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group"""

        test_res = self.testing_handler(self.args, config['model_id'])
        test_res = {'executorId': self.this_rank, 'results': test_res, 'clusterId': config['model_id']}

        # Report execution completion information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id = self.executor_id, executor_id = self.executor_id,
                event = events.MODEL_TEST, status = True, msg = None,
                meta_result = None, data_result = self.serialize_response(test_res)
            )
        )
        self.dispatch_worker_events(response)


    def report_executor_info_handler(self):
        """Return the statistics of training dataset"""
        return self.training_sets.getSize()

    
    def update_model_handler(self, model, model_id):
        """Update the model copy on this executor"""
        self.models[model_id] = model
        self.round[model_id] += 1

        # Dump latest model to disk
        with open(self.temp_model_path + '_' + str(model_id), 'wb') as model_out:
            pickle.dump(self.models[model_id], model_out)


    def load_global_model(self, model_id):
        # load last global model
        with open(self.temp_model_path + '_' + str(model_id), 'rb') as model_in:
            model = pickle.load(model_in)
        assert(model is not None)
        return model


    def override_conf(self, config):
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)


    def get_client_trainer(self, conf):
        """Developer can redefine to this function to customize the training:
           API:
            - train(client_data=client_data, model=client_model, conf=conf)
        """
        return Customized_Client(conf)


    def training_handler(self, clientId, conf, model=None):
        """Train model given client ids"""

        # load last global model
        assert(model is None)
        model_id = conf.model_id
        client_model = self.load_global_model(model_id) if model is None else model

        conf.clientId, conf.device = clientId, self.device
        # disable nlp and rl temporarily
        # conf.tokenizer = tokenizer
        conf.tokenizer = None

        client_data = select_dataset(clientId, self.training_sets, batch_size=conf.batch_size, args = self.args, collate_fn=self.collate_fn)

        client = self.get_client_trainer(conf)
        train_res, model = client.train(client_data=client_data, model=client_model, conf=conf, dry_run=args.dry_train)

        return train_res, model


    def validation_handler(self, clientId, model):
        """Validate model loss given client ids"""

        # not support rl or nlp tasks
        client_data = select_dataset(clientId, self.val_sets, batch_size=self.args.batch_size // 5, args=self.args, isTest=False, collate_fn=self.collate_fn)
        criterion = torch.nn.CrossEntropyLoss().to(device=self.device)
        val_res = validate_model(clientId, model, client_data, self.device, criterion=criterion, dry_run=args.dry_validate)
        
        val_loss, acc, acc_5, valResults = val_res
        logging.info("At training round {}, client {} ({} labels) has val_loss {}, val_accuracy {:.2f}%, val_5_accuracy {:.2f}% \n"\
                     .format(self.round, clientId, len(self.training_sets.client_label_cnt[clientId]), val_loss, acc, acc_5))

        return valResults


    def testing_handler(self, args, model_id):
        """Test model"""
        evalStart = time.time()
        device = self.device

        data_loader = select_dataset(self.this_rank, self.testing_sets, batch_size=args.test_bsz, args = self.args, isTest=True, collate_fn=self.collate_fn)

        criterion = torch.nn.CrossEntropyLoss().to(device=device)

        model = self.models[model_id]

        if len(self.klayers_outputs) != self.sploss_gap:
            logging.info(f"sploss gap: {self.sploss_gap}, current length: {len(self.klayers_outputs)}")
            test_res = test_model(self.this_rank, model, data_loader, device=device, criterion=criterion, dry_run=args.dry_test, layers_names=self.archi_manager.get_trainable_layer_names())
            test_loss, acc, acc_5, testResults, layers_outputs, _ = test_res
            self.klayers_outputs.append(layers_outputs)
            logging.info("Cluster: {}, After aggregation epoch {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                        .format(model_id, self.round, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))
        else:
            test_res = test_model(self.this_rank, model, data_loader, device=device, criterion=criterion, reference=self.klayers_outputs[0], dry_run=args.dry_test, layers_names=self.archi_manager.get_trainable_layer_names())
            test_loss, acc, acc_5, testResults, layers_outputs, sploss = test_res
            self.klayers_outputs.append(layers_outputs)
            self.klayers_outputs.pop(0)
            logging.info("Cluster: {}, After aggregation epoch {}, CumulTime {}, eval_time {}, sploss {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                        .format(model_id, self.round, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), sploss, test_loss, acc*100., acc_5*100.))
        gc.collect()

        return testResults 


    def client_register(self):
        """Register the executor information to the aggregator"""
        start_time = time.time()
        while time.time() - start_time < 180:
            try:
                response = self.aggregator_communicator.stub.CLIENT_REGISTER(
                    job_api_pb2.RegisterRequest(
                        client_id = self.executor_id,
                        executor_id = self.executor_id,
                        executor_info = self.serialize_response(self.report_executor_info_handler())
                    )
                )
                self.dispatch_worker_events(response)
                break
            except Exception as e:
                logging.warning(f"Failed to connect to aggregator {e}. Will retry in 5 sec.")
                time.sleep(5)


    def client_ping(self):
        """Ping the aggregator for new task"""
        response = self.aggregator_communicator.stub.CLIENT_PING(job_api_pb2.PingRequest(
            client_id = self.executor_id,
            executor_id = self.executor_id
        ))
        self.dispatch_worker_events(response)


    def event_monitor(self):
        """Activate event handler once receiving new message"""
        logging.info("Start monitoring events ...")
        self.client_register()

        while self.received_stop_request == False:
            if len(self.event_queue) > 0:
                request = self.event_queue.popleft()
                current_event = request.event

                if current_event == events.CLIENT_TRAIN:
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    train_config['model'] = train_model
                    train_config['client_id'] = int(train_config['client_id'])
                    client_id, train_res = self.Train(train_config)

                    # Upload model updates
                    _ = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                        job_api_pb2.CompleteRequest(client_id = str(client_id), executor_id = self.executor_id,
                        event = events.UPLOAD_MODEL, status = True, msg = None,
                        meta_result = None, data_result = self.serialize_response(train_res)
                    ))

                elif current_event == events.MODEL_TEST:
                    self.Test(self.deserialize_response(request.meta))

                elif current_event == events.UPDATE_MODEL:
                    broadcast_config = self.deserialize_response(request.data)
                    model_id = self.deserialize_response(request.meta)
                    self.UpdateModel(broadcast_config, model_id)

                elif current_event == events.SHUT_DOWN:
                    self.Stop()

                elif current_event == events.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                self.client_ping()



if __name__ == "__main__":
    executor = Customized_Executor(args)
    executor.run()