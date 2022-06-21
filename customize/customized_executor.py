import logging
import torch
import gc
import time

from fedscale.core.executor import Executor
from fedscale.core.rlclient import RLClient
from fedscale.core import events
import fedscale.core.job_api_pb2 as job_api_pb2
from fedscale.core.fllibs import init_model

from customized_utils.architecture_manager import Architecture_Manager
from customized_arg_parser import args
from customized_fllib import init_dataset
from customized_utils.customized_utils_models import validate_model, test_model
from customized_client import Customized_Client
from customized_utils.customized_divide_data import Customized_DataPartitioner, select_dataset
from config import cfg

class Customized_Executor(Executor):
    def __init__(self, args):
        super().__init__(args)
        self.klayers_outputs = []
        self.sploss_gap = cfg['sploss_gap']


    def run(self):
        self.setup_env()
        self.model = init_model()
        self.model = self.model.to(device=self.device)
        self.training_sets, self.testing_sets, self.val_sets = self.init_data()
        self.setup_communication()
        dummy_input = torch.randn(10, 3, 32, 32, device=self.device)
        self.archi_manager = Architecture_Manager(dummy_input, 'customize_clients.onnx')
        self.archi_manager.parse_model(self.model)
        self.event_monitor()
    
    def init_data(self):
        """Return the training, testing and validation dataset"""
        train_dataset, test_dataset, val_dataset = init_dataset()
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

    def get_client_trainer(self, conf):
        """Developer can redefine to this function to customize the training:
           API:
            - train(client_data=client_data, model=client_model, conf=conf)
        """
        return Customized_Client(conf)


    def training_handler(self, clientId, conf, model=None):
        """Train model given client ids"""

        # load last global model
        client_model = self.load_global_model() if model is None else model

        conf.clientId, conf.device = clientId, self.device
        # conf.tokenizer = tokenizer
        conf.tokenizer = None
        if args.task == "rl":
            client_data = self.training_sets
            client = RLClient(conf)
            train_res = client.train(client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets, batch_size=conf.batch_size, args = self.args, collate_fn=self.collate_fn)

            client = self.get_client_trainer(conf)
            train_res, model = client.train(client_data=client_data, model=client_model, conf=conf, dry_run=args.dry_train)

        return train_res, model

    def Test(self, config):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group"""

        test_res = self.testing_handler(self.args, config['model'])
        test_res = {'executorId': self.this_rank, 'results': test_res}

        # Report execution completion information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id = self.executor_id, executor_id = self.executor_id,
                event = events.MODEL_TEST, status = True, msg = None,
                meta_result = None, data_result = self.serialize_response(test_res)
            )
        )
        self.dispatch_worker_events(response)

    def testing_handler(self, args, models):
        """Test model"""
        evalStart = time.time()
        device = self.device

        data_loader = select_dataset(self.this_rank, self.testing_sets, batch_size=args.test_bsz, args = self.args, isTest=True, collate_fn=self.collate_fn)

        criterion = torch.nn.CrossEntropyLoss().to(device=device)

        for i, model in enumerate(models[:-1]):
            test_res = test_model(self.this_rank, model, data_loader, device=device, criterion=criterion, dry_run=args.dry_test)
            test_loss, acc, acc_5, testResults, _, _ = test_res
            logging.info("Cluster: {}, After aggregation epoch {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                        .format(i, self.round, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))


        if len(self.klayers_outputs) != self.sploss_gap:
            logging.info(f"sploss gap: {self.sploss_gap}, current length: {len(self.klayers_outputs)}")
            test_res = test_model(self.this_rank, models[-1], data_loader, device=device, criterion=criterion, dry_run=args.dry_test, layers_names=self.archi_manager.get_trainable_layer_names())
            test_loss, acc, acc_5, testResults, layers_outputs, _ = test_res
            self.klayers_outputs.append(layers_outputs)
            logging.info("Cluster: {}, After aggregation epoch {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                        .format(len(models), self.round, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))
        else:
            test_res = test_model(self.this_rank, models[-1], data_loader, device=device, criterion=criterion, reference=self.klayers_outputs[0], dry_run=args.dry_test, layers_names=self.archi_manager.get_trainable_layer_names())
            test_loss, acc, acc_5, testResults, layers_outputs, sploss = test_res
            self.klayers_outputs.append(layers_outputs)
            self.klayers_outputs.pop(0)
            logging.info("Cluster: {}, After aggregation epoch {}, CumulTime {}, eval_time {}, sploss {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                        .format(len(models), self.round, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), sploss, test_loss, acc*100., acc_5*100.))
        gc.collect()

        return testResults 

    def Train(self, config):
        """Integrate validation into training"""
        client_id, train_config = config['client_id'], config['task_config']

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

if __name__ == "__main__":
    executor = Customized_Executor(args)
    executor.run()