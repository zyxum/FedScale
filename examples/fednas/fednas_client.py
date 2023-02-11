from fedscale.cloud.execution.client import Client
from darts.architect import Architect

import logging
import torch.nn as nn
import torch
import math

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class FedNAS_Client(Client):
    def local_search(self, train_data, test_data, model, architect, criterion, optimizer, device, conf, client_id):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        loss = None
        for step, (input, target) in enumerate(train_data):

            n = input.size(0)

            input = input.to(device)
            target = target.to(device)

            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(test_data))
            input_search = input_search.to(device)
            target_search = target_search.to(device)

            architect.step_v2(input, target, input_search, target_search, conf.lambda_train_regularizer,
                              conf.lambda_valid_regularizer)

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            parameters = model.arch_parameters()
            nn.utils.clip_grad_norm_(parameters, conf.grad_clip)
            optimizer.step()

            # logging.info("step %d. update weight by SGD. FINISH\n" % step)
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # torch.cuda.empty_cache()

            if step % conf.report_freq == 0:
                logging.info('client_index = %d, search %03d %e %f %f', client_id,
                             step, objs.avg, top1.avg, top5.avg)

        return top1.avg / 100.0, objs.avg / 100.0, loss

    def search(self, train_data, test_data, model: nn.Module, conf):
        clientId = conf.clientId
        device = conf.device
        model.to(device)
        model.train()

        arch_parameters = model.arch_parameters()
        arch_params = list(map(id, arch_parameters))

        parameters = model.parameters()
        weight_params = filter(lambda p: id(p) not in arch_params,
                               parameters)

        optimizer = torch.optim.SGD(
            weight_params,  # model.parameters(),
            conf.learning_rate,
            momentum=conf.momentum,
            weight_decay=conf.weight_decay)

        criterion = torch.nn.CrossEntropyLoss().to(device=conf.device)

        architect = Architect(model, criterion, conf, device)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, conf.local_step, eta_min=conf.min_learning_rate)

        local_avg_train_acc = []
        local_avg_train_loss = []
        for epoch in range(conf.local_step):
            # training
            train_acc, train_obj, train_loss = self.local_search(train_data, test_data,
                                                                 model, architect, criterion,
                                                                 optimizer, device, conf, clientId)
            logging.info('client_idx = %d, epoch = %d, local search_acc %f' % (clientId, epoch, train_acc))
            local_avg_train_acc.append(train_acc)
            local_avg_train_loss.append(train_loss)

            scheduler.step()
            lr = scheduler.get_lr()
            logging.info('client_idx = %d, epoch %d lr %e' % (clientId, epoch, lr))

        alphas = model.cpu().arch_parameters()

        loss_square = sum([l**2 for l in local_avg_train_loss]) / float(len(local_avg_train_loss))

        state_dicts = model.state_dict()
        model_param = {p: state_dicts[p].data.cpu().numpy()
                       for p in state_dicts}

        results = {'clientId': clientId, 'moving_loss': sum(local_avg_train_loss) / len(local_avg_train_loss),
                   'trained_size': len(train_data),
                   'success': True, 'utility': math.sqrt(loss_square)*float(len(train_data)),
                   'update_weight': model_param, "arch": alphas
                   }

        logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")

        return results

    def local_train(self, train_data, model, criterion, optimizer, device, conf):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        loss = None
        for step, (input, target) in enumerate(train_data):
            # logging.info("epoch %d, step %d START" % (epoch, step))
            model.train()
            n = input.size(0)

            input = input.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            loss.backward()
            parameters = model.parameters()
            nn.utils.clip_grad_norm_(parameters, conf.grad_clip)
            optimizer.step()
            # logging.info("step %d. update weight by SGD. FINISH\n" % step)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        return top1.avg, objs.avg, loss

    def train(self, train_data, model: nn.Module, conf):
        device = conf.device
        clientId = conf.clientId
        model.to(device)
        model.train()

        parameters = model.parameters()

        optimizer = torch.optim.SGD(
            parameters,  # model.parameters(),
            conf.learning_rate,
            conf.momentum,
            weight_decay=conf.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, conf.local_steps, eta_min=conf.learning_rate_min)

        criterion = torch.nn.CrossEntropyLoss().to(device=conf.device)

        local_avg_train_acc = []
        local_avg_train_loss = []
        for epoch in range(conf.local_steps):
            # training
            train_acc, train_obj, train_loss = self.local_train(train_data,
                                                                model, criterion,
                                                                optimizer, device, conf)
            local_avg_train_acc.append(train_acc)
            local_avg_train_loss.append(train_loss)

            scheduler.step()

        loss_square = sum([l**2 for l in local_avg_train_loss]) / len(local_avg_train_loss)

        state_dicts = model.state_dict()
        model_param = {p: state_dicts[p].data.cpu().numpy()
                       for p in state_dicts}

        results = {'clientId': clientId, 'moving_loss': sum(local_avg_train_loss) / len(local_avg_train_loss),
                   'trained_size': len(train_data),
                   'success': True, 'utility': math.sqrt(loss_square)*float(len(train_data)),
                   'update_weight': model_param, "arch": None
                   }

        logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")

        return results