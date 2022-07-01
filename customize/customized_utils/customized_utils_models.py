from torch.autograd import Variable
import torch.nn as nn
import torch, logging
import torch.nn.functional as F
import logging
from customized_utils.net2net import get_model_layer

from fedscale.core.utils.model_test_module import accuracy

def layer_batch_norm(x):
    x = torch.reshape(x, (x.shape[0], -1))
    x = torch.matmul(x, torch.t(x))
    x = F.normalize(x, 2, 1)
    return x


def validate_model(clientId, model, val_data, device='cpu', criterion=nn.NLLLoss(), tokenizer=None, dry_run: bool=False):
    val_loss = 0
    correct = 0
    top_5 = 0
    val_len = 0

    model = model.to(device=device)
    model.eval()
    
    # only support image classification tasks
    with torch.no_grad():
        for data, target in val_data:
            try:
                data, target = Variable(data).to(device=device), Variable(target).to(device=device)

                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.data.item()
                acc = accuracy(output, target, topk=(1,5))

                correct += acc[0].item()
                top_5 += acc[1].item()
            
                if dry_run:
                    break
            except Exception as ex:
                logging.info(f"Validation of {clientId} failed as {ex}")
                break
            val_len += len(target)

    val_len = max(val_len, 1)
    # loss function averages over batch size
    val_loss /= len(val_data)

    sum_loss = val_loss * val_len

    acc = round(correct / val_len, 4)
    acc_5 = round(top_5 / val_len, 4)
    val_loss = round(val_loss, 4)

    logging.info('Client {}: Validation set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
          .format(clientId, val_loss, correct, len(val_data.dataset), acc, acc_5))
    
    valRes = {'top_1':correct, 'top_5':top_5, 'val_loss':val_loss, 'val_len': val_len}

    # exhaust dataloader
    if dry_run:
        for data, target in val_data:
            continue
    return val_loss, acc, acc_5, valRes

def test_model(rank, model, test_data, device='cpu', criterion=nn.NLLLoss(), reference=[], dry_run: bool=False, layers_names=[]):

    test_loss = 0
    correct = 0
    top_5 = 0

    test_len = 0

    model = model.to(device=device) # load by pickle
    model.eval()
    sploss = {}

    layers_outputs = []
    layer_output = {}
    hook_handles = []
    
    def get_activation(name):
        def hook(model, input, output):
            layer_output[name] = output.detach().cpu()
        return hook

    # register hooks
    for layer_name in layers_names:
        layer = get_model_layer(model, layer_name)
        hook_handles.append(
            layer.register_forward_hook(get_activation(layer_name))
        )

    # count = 0 # only for debug
    with torch.no_grad():
        for data, target in test_data:
            try:
                data, target = Variable(data).to(device=device), Variable(target).to(device=device)

                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.data.item()  # Variable.data
                acc = accuracy(output, target, topk=(1, 5))

                correct += acc[0].item()
                top_5 += acc[1].item()

                # transform and record layer output
                for key in layer_output.keys():
                    layer_output[key] = layer_batch_norm(layer_output[key])
                layers_outputs.append(layer_output)

                if dry_run:
                    break
                    
            except Exception as ex:
                logging.info(f"Testing of failed as {ex}")
                break
            test_len += len(target)
            # only for debug
            # count += 1
            # if count > 5:
            #     break

    if len(reference) != 0:
        for i, layer_output in enumerate(layers_outputs):
            for key in layer_output.keys():
                if key not in sploss.keys():
                    sploss[key] = 0
                sploss[key] += torch.norm(layer_output[key] - reference[i][key]) ** 2 / (layer_output[key].shape[0] ** 2)
    

    test_len = max(test_len, 1)
    # loss function averages over batch size
    test_loss /= len(test_data)

    sum_loss = test_loss * test_len

    # in NLP, we care about the perplexity of the model
    acc = round(correct / test_len, 4)
    acc_5 = round(top_5 / test_len, 4)
    test_loss = round(test_loss, 4)

    logging.info('Rank {}: Test set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
          .format(rank, test_loss, correct, len(test_data.dataset), acc, acc_5))

    testRes = {'top_1':correct, 'top_5':top_5, 'test_loss':sum_loss, 'sp_loss':sploss, 'test_len':test_len}

    # remove hooks
    for handle in hook_handles:
        handle.remove()

    # exhaust dataloader
    # if dry_run:
    #     for data, target in test_data:
    #         continue

    logging.info("finish testing")
    return test_loss, acc, acc_5, testRes, layers_outputs, sploss