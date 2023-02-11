from fedscale.cloud.execution.executor import Executor
from darts.model_search import Network
from fedscale.cloud.fllibs import outputClass, tokenizer, select_dataset, DataPartitioner, init_dataset
from fedscale.cloud.execution.data_processor import collate, voice_collate_fn
import fednas_config_parser as parser
from fednas_client import FedNAS_Client

import torch.nn as nn
import pickle
from copy import deepcopy
import logging



class FedNAS_Executor(Executor):
    def __init__(self, args):
        super().__init__(args)
        self.stage = "search"

    def init_model(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        model = Network(self.args.init_channels, num_classes=outputClass[self.args.data_set],
                        layers=self.args.layers, criterion=criterion, device=self.device)
        self.model = model

    def init_data(self):
        """Return the training and testing dataset

        Returns:
            Tuple of DataPartitioner class: The partioned dataset class for training and testing

        """
        train_dataset, test_dataset = init_dataset()
        if self.task == "rl":
            return train_dataset, test_dataset
        # load data partitioner (entire_train_data)
        logging.info("Data partitioner starts ...")

        training_sets = DataPartitioner(
            data=train_dataset, args=self.args, numOfClass=self.args.num_class)
        training_sets.partition_data_helper(
            num_clients=self.args.num_participants, data_map_file=self.args.data_map_file)

        testing_sets = DataPartitioner(
            data=test_dataset, args=self.args, numOfClass=self.args.num_class, isTest=True)
        testing_sets.partition_data_helper(num_clients=self.args.num_participants,
                                           data_map_file=self.args.test_data_map_file)

        logging.info("Data partitioner completes ...")

        if self.task == 'nlp':
            self.collate_fn = collate
        elif self.task == 'voice':
            self.collate_fn = voice_collate_fn

        return training_sets, testing_sets

    def run(self):
        """Start running the executor by setting up execution and communication environment, and monitoring the grpc message.
        """
        self.setup_env()
        self.training_sets, self.testing_sets = self.init_data()
        self.init_model()
        self.setup_communication()
        self.event_monitor()

    def load_global_model(self):
        """ Load last global model

        Returns:
            PyTorch or TensorFlow model: The lastest global model

        """
        with open(self.temp_model_path, 'rb') as model_in:
            model_package = pickle.load(model_in)
        if self.stage == "train":
            self.model = model_package
        else:
            self.model.load_state_dict(model_package["model_params"])
            for a_g, model_arch in zip(model_package["arch_params"], self.model.arch_parameters()):
                model_arch.data.copy_(a_g.data)

    def update_model_handler(self, model):
        """Update the model copy on this executor

        Args:
            config (PyTorch or TensorFlow model): The broadcasted global model

        """
        self.round += 1
        if self.round == self.args.search_round:
            self.stage = "train"

        # Dump latest model to disk
        with open(self.temp_model_path, 'wb') as model_out:
            pickle.dump(model, model_out)

    def get_client_trainer(self, conf):
        """A abstract base class for client with training handler, developer can redefine to this function to customize the client training:

        Args:
            config (dictionary): The client runtime config.

        Returns:
            Client: A abstract base client class with runtime config conf.

        """
        return FedNAS_Client(conf)

    def training_handler(self, clientId, conf, model=None):
        """Train model given client id

        Args:
            clientId (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result

        """
        # load last global model
        self.load_global_model()
        client_model = deepcopy(self.model)

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer
        train_data = select_dataset(clientId, self.training_sets,
                                     batch_size=conf.batch_size, args=self.args,
                                     collate_fn=self.collate_fn
                                     )

        test_data = select_dataset(clientId, self.testing_sets,
                                   batch_size=conf.batch_size, args=self.args,
                                   isTest=True, collate_fn=self.collate_fn)

        client = self.get_client_trainer(conf)

        if self.stage == "search":
            res = client.search(train_data, test_data, client_model, conf)
        else:
            res = client.train(train_data, test_data, client_model, conf)

        return res

if __name__ == "__main__":
    executor = FedNAS_Executor(parser.args)
    executor.run()