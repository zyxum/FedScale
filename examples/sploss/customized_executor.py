from fedscale.core.executor import Executor
from fedscale.core.utils.utils_model import accuracy
from fedscale.core.arg_parser import args
import grpc
import fedscale.core.job_api_pb2_grpc as job_api_pb2_grpc
from concurrent import futures
from customized_dataloader import select_dataset
import time
import torch
import torch.nn as nn
import logging
import gc
from config import cfg

def test_model_sploss(rank, model, data_loader, device='cpu', criterion=nn.NLLLoss(), reference=[]):
    test_loss = 0
    correct = 0
    top_5 = 0

    test_len = 0

    model = model.to(device=device) # load by pickle
    model.eval()
    sploss_list = []
    sploss = []

    with torch.no_grad():
        for data, target in data_loader:
            # try:
            data = data.to(device=device)
            target = target.to(device=device)
            output, sploss_temp = model(data, False)
            loss = criterion(output, target)
            test_loss += loss.data.item()
            acc = accuracy(output, target, topk=(1,5))
            correct += acc[0].item()
            top_5 += acc[1].item()
            sploss_list.append(sploss_temp)
            # except Exception as ex:
            #     logging.info(f"Testing of failed as {ex}")
            #     break
            test_len += len(target)

    if len(reference) != 0:
        for i, batch in enumerate(sploss_list):
            for j, layer in enumerate(batch):
                layer_loss = torch.norm(layer - reference[i][j]) ** 2 / (layer.shape[0] ** 2)
                if i == 0:
                    sploss.append(layer_loss)
                else:
                    sploss[j] += layer_loss
    
    test_len = max(test_len, 1)
    test_loss /= len(data_loader)
    sum_loss = test_loss * test_len
    acc = round(correct / test_len, 4)
    acc_5 = round(top_5 / test_len, 4)
    test_loss = round(test_loss, 4)

    logging.info('Rank {}: Test set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
          .format(rank, test_loss, correct, len(data_loader.dataset), acc, acc_5))

    testRes = {'top_1':correct, 'top_5':top_5, 'test_loss':sum_loss, 'sp_loss':torch.tensor(sploss), 'test_len':test_len}
    
    return test_loss, acc, acc_5, testRes, sploss_list, sploss

class Customized_Executor(Executor):
    def __init__(self, args):
        super().__init__(args)
        self.ksploss = []
        self.sploss_gap = cfg['sploss_gap']

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages."""

        logging.info(f"Connecting to Coordinator ({args.ps_ip}) for control plane communication ...")

        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', 90000000),
                ('grpc.max_receive_message_length', 90000000),
            ],
        )
        job_api_pb2_grpc.add_JobServiceServicer_to_server(self, self.grpc_server)
        port = '[::]:{}'.format(self.args.base_port + self.this_rank)
        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()
        logging.info(f'Started GRPC server at {port} for control plane')

    def testing_handler(self, args):
        """Test model"""
        evalStart = time.time()
        device = self.device
        model = self.load_global_model()

        data_loader = select_dataset(self.this_rank, self.testing_sets, batch_size=args.test_bsz, args = self.args, isTest=True, collate_fn=self.collate_fn)

        criterion = torch.nn.CrossEntropyLoss().to(device=device)

        if len(self.ksploss) != self.sploss_gap:
            test_res = test_model_sploss(self.this_rank, model, data_loader, device=device, criterion=criterion)
            test_loss, acc, acc_5, testResults, sploss_list, _ = test_res
            self.ksploss.append(sploss_list)
            logging.info("After aggregation epoch {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                        .format(self.epoch, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))
        else:
            test_res = test_model_sploss(self.this_rank, model, data_loader, device=device, criterion=criterion, reference=self.ksploss[0])
            test_loss, acc, acc_5, testResults, sploss_list, sploss = test_res
            self.ksploss.append(sploss_list)
            self.ksploss.pop(0)
            logging.info("After aggregation epoch {}, CumulTime {}, eval_time {}, sploss {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                        .format(self.epoch, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), sploss, test_loss, acc*100., acc_5*100.))

        gc.collect()

        return testResults

if __name__ == "__main__":
    executor = Customized_Executor(args)
    executor.run()