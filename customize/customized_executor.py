import logging
import torch
from fedscale.core.executor import Executor
from fedscale.core.utils.divide_data import DataPartitioner, select_dataset
from fedscale.core.rlclient import RLClient
from fedscale.core import events
import fedscale.core.job_api_pb2 as job_api_pb2

from customized_arg_parser import args
from customized_fllib import init_dataset
from customized_utils.customized_utils_models import validate_model
from customized_client import Customized_Client
from customized_utils.customized_divide_data import Customized_DataPartitioner
class Customized_Executor(Executor):

    def run(self):
        self.setup_env()
        self.model = self.init_model()
        self.model = self.model.to(device=self.device)
        self.training_sets, self.testing_sets, self.val_sets = self.init_data()
        self.setup_communication()
        self.event_monitor()
    
    def init_data(self):
        """Return the training, testing and validation dataset"""
        train_dataset, test_dataset, val_dataset = init_dataset()
        logging.info("Data partitioner starts ...")

        training_sets = Customized_DataPartitioner(data=train_dataset, args = self.args, numOfClass=self.args.num_class)
        unique_clientId = training_sets.partition_data_helper(num_clients=self.args.total_worker, data_map_file=self.args.data_map_file)

        testing_sets = Customized_DataPartitioner(data=test_dataset, args = self.args, numOfClass=self.args.num_class, isTest=True)
        testing_sets.partition_data_helper(num_clients=self.num_executors)

        val_sets = Customized_DataPartitioner(data=val_dataset, args = self.args, numOfClass=self.args.num_class)
        val_sets.partition_data_helper(num_clients=self.args.total_worker, data_map_file=self.args.val_data_map_file, unique_clientId=unique_clientId)

        logging.info("Data partitioner competes ...")

        # not support nlp or voice tasks

        return training_sets, testing_sets, val_sets

    def validation_handler(self, clientId, model):
        """Validate model loss given client ids"""

        # not support rl or nlp tasks
        client_data = select_dataset(clientId, self.val_sets, batch_size=self.args.test_bsz, args=self.args, isTest=True, collate_fn=self.collate_fn)
        criterion = torch.nn.CrossEntropyLoss().to(device=self.device)
        val_res = validate_model(clientId, model, client_data, self.device, criterion=criterion)
        
        val_loss, acc, acc_5, valResults = val_res
        logging.info("At training round {}, client {} has val_loss {}, val_accuracy {:.2f}%, val_5_accuracy {:.2f}% \n"\
                     .format(self.round, clientId, val_loss, acc, acc_5))

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
            train_res, model = client.train(client_data=client_data, model=client_model, conf=conf)

        return train_res, model

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