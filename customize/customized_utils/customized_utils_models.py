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