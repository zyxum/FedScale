from fedscale.cloud.execution.executor import Executor
from fedscale.cloud.execution.rlclient import RLClient
from fedscale.dataloaders.divide_data import select_dataset
from fedscale.cloud.fllibs import tokenizer

from fjord_client import FjORD_Client

import pickle

class FjORD_Executor(Executor):

    def load_global_model(self, tier):
        """ Load last global model

        Returns:
            PyTorch or TensorFlow model: The lastest global model

        """
        with open(self.temp_model_path, 'rb') as model_in:
            model = pickle.load(model_in)
        return model[tier]

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
        tier = int(conf.p)
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
                client_data=client_data, model=client_model, conf=conf)

            train_res["tier"] = tier

        return train_res