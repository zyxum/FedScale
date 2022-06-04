from torch.autograd import Variable
import torch.nn as nn
import torch, logging

from fedscale.core.utils.utils_model import accuracy
def validate_model(clientId, model, val_data, device='cpu', criterion=nn.NLLLoss(), tokenizer=None):
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
    
    valRes = {'top_1':correct, 'top_5':top_5, 'val_loss':sum_loss, 'val_len': val_len}

    return val_loss, acc, acc_5, valRes

def test_model(rank, model, test_data, device='cpu', criterion=nn.NLLLoss(), reference=[]):

    test_loss = 0
    correct = 0
    top_5 = 0

    test_len = 0

    model = model.to(device=device) # load by pickle
    model.eval()
    sploss_list = []
    sploss = []

    with torch.no_grad():
        for data, target in test_data:
            try:
                data, target = Variable(data).to(device=device), Variable(target).to(device=device)

                output, sploss_temp = model(data, False)
                loss = criterion(output, target)
                
                test_loss += loss.data.item()  # Variable.data
                acc = accuracy(output, target, topk=(1, 5))

                correct += acc[0].item()
                top_5 += acc[1].item()
                sploss_list.append(sploss_temp)
        
            except Exception as ex:
                logging.info(f"Testing of failed as {ex}")
                break
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
    # loss function averages over batch size
    test_loss /= len(test_data)

    sum_loss = test_loss * test_len

    # in NLP, we care about the perplexity of the model
    acc = round(correct / test_len, 4)
    acc_5 = round(top_5 / test_len, 4)
    test_loss = round(test_loss, 4)

    logging.info('Rank {}: Test set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
          .format(rank, test_loss, correct, len(test_data.dataset), acc, acc_5))

    testRes = {'top_1':correct, 'top_5':top_5, 'test_loss':sum_loss, 'sp_loss':torch.tensor(sploss), 'test_len':test_len}

    return test_loss, acc, acc_5, testRes, sploss_list, sploss