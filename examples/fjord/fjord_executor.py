from fedscale.cloud.execution.executor import Executor
from fedscale.cloud.execution.rlclient import RLClient
from fedscale.dataloaders.divide_data import select_dataset
from fedscale.cloud.fllibs import tokenizer, DataPartitioner, init_dataset
from fedscale.utils.model_test_module import test_model
from fedscale.cloud import commons
import fedscale.cloud.channels.job_api_pb2 as job_api_pb2

from fjord_client import FjORD_Client
import fjord_config_parser as parser

import pickle
import time
import logging
import torch
import gc

class FjORD_Executor(Executor):

    def load_global_model(self, tier):
        """ Load last global model

        Returns:
            PyTorch or TensorFlow model: The lastest global model

        """
        with open(self.temp_model_path, 'rb') as model_in:
            model = pickle.load(model_in)
        return model[tier]

    def load_global_model_list(self):
        with open(self.temp_model_path, 'rb') as model_in:
            models = pickle.load(model_in)
        return models

    def get_client_trainer(self, conf):
        """A abstract base class for client with training handler, developer can redefine to this function to customize the client training:

        Args:
            config (dictionary): The client runtime config.

        Returns:
            Client: A abstract base client class with runtime config conf.

        """
        return FjORD_Client(conf)

    def training_handler(self, clientId, conf, model=None):
        """Train model given client id

                Args:
                    clientId (int): The client id.
                    conf (dictionary): The client runtime config.

                Returns:
                    dictionary: The train result

                """
        # load last global model
        tier = int(conf.tier)
        client_model = self.load_global_model(tier) if model is None else model

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer
        if self.args.task == "rl":
            client_data = self.training_sets
            client = RLClient(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets,
                                         batch_size=conf.batch_size, args=self.args,
                                         collate_fn=self.collate_fn
                                         )

            client = self.get_client_trainer(conf)
            train_res = client.train(
                client_data=client_data, max_model=client_model, conf=conf)

            train_res["tier"] = tier

        return train_res


    def testing_handler(self, args, config=None):
        """Test model
        
        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
            config (dictionary): Variable arguments from coordinator.
        Returns:
            dictionary: The test result

        """
        evalStart = time.time()
        device = self.device
        if self.task == 'rl':
            client = RLClient(args)
            test_res = client.test(args, self.this_rank, model, device=device)
            _, _, _, testResults = test_res
        else:
            models = self.load_global_model_list()
            for tier in models:
                model = models[tier]
                data_loader = select_dataset(self.this_rank, self.testing_sets,
                                            batch_size=args.test_bsz, args=args,
                                            isTest=True, collate_fn=self.collate_fn
                                            )

                criterion = torch.nn.CrossEntropyLoss().to(device=device)

                if self.args.engine == commons.PYTORCH:
                    test_res = test_model(self.this_rank, model, data_loader,
                                        device=device, criterion=criterion, tokenizer=tokenizer)
                else:
                    raise Exception(f"Need customized implementation for model testing in {self.args.engine} engine")

                test_loss, acc, acc_5, testResults = test_res
                logging.info("Model Tier {}: After aggregation round {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                            .format(tier, self.round, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))


        gc.collect()

        return testResults


if __name__ == "__main__":
    executor = FjORD_Executor(parser.args)
    executor.run()