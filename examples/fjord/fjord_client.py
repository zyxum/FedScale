from fedscale.cloud.execution.client import Client
from fedscale.cloud.execution.optimizers import ClientOptimizer
from fjord_utils import sample_subnetwork, load_sub_model
from fedscale.dataloaders.nlp import mask_tokens

import logging
import random
import torch
from torch.autograd import Variable
import math

class FjORD_Client(Client):

    def init_task(self, conf):
        self.epoch_train_loss = 1e-4
        self.completed_steps = 0
        self.loss_squre = 0
        self.p_max = conf.p
        self.p_pool = conf.p_pool

    def get_optimizer(self, model, conf):
        optimizer = None
        if conf.task == "detection":
            lr = conf.learning_rate
            params = []
            for key, value in dict(model.named_parameters()).items():
                if value.requires_grad:
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': lr * (conf.TRAIN.DOUBLE_BIAS + 1),
                                    'weight_decay': conf.TRAIN.BIAS_DECAY and conf.TRAIN.WEIGHT_DECAY or 0}]
                    else:
                        params += [{'params': [value], 'lr': lr,
                                    'weight_decay': conf.TRAIN.WEIGHT_DECAY}]
            optimizer = torch.optim.SGD(params, momentum=conf.TRAIN.MOMENTUM)

        elif conf.task == 'nlp':

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": conf.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            # Bert pre-training setup
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters, lr=conf.learning_rate, weight_decay=1e-2)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=conf.learning_rate, weight_decay=5e-4)
        return optimizer

    def train_step_kd(self, client_data, conf, max_model):
        for data_pair in client_data:
            # data&target preprocess
            if conf.task == 'nlp':
                (data, _) = data_pair
                data, target = mask_tokens(
                    data, tokenizer, conf, device=conf.device)
            elif conf.task == 'voice':
                (data, target, input_percentages,
                 target_sizes), _ = data_pair
                input_sizes = input_percentages.mul_(
                    int(data.size(3))).int()
            elif conf.task == 'detection':
                temp_data = data_pair
                target = temp_data[4]
                data = temp_data[0:4]
            else:
                (data, target) = data_pair
            # data preprocess
            if conf.task == "detection":
                self.im_data.resize_(data[0].size()).copy_(data[0])
                self.im_info.resize_(data[1].size()).copy_(data[1])
                self.gt_boxes.resize_(data[2].size()).copy_(data[2])
                self.num_boxes.resize_(data[3].size()).copy_(data[3])
            elif conf.task == 'speech':
                data = torch.unsqueeze(data, 1).to(device=conf.device)
            elif conf.task == 'text_clf' and conf.model == 'albert-base-v2':
                (data, masks) = data
                data, masks = Variable(data).to(
                    device=conf.device), Variable(masks).to(device=conf.device)


            else:
                data = Variable(data).to(device=conf.device)

            target = Variable(target).to(device=conf.device)

            # sample a submodel
            valid_p = [p_value for p_value in conf.p_pool if p_value <= conf.p_max]
            p_value = random.choice(valid_p)
            ratio = p_value / conf.p_max
            sub_model = sample_subnetwork(max_model, ratio)

            # set up sub_model
            sub_model.to(device=conf.device)

            # set up max_model
            max_model.to(device=conf.device)
            max_model.train()
            max_optimizer = self.get_optimizer(max_model, conf)

            # criterion
            CE = torch.nn.CrossEntropyLoss(reduction='none').to(device=conf.device)
            KL = torch.nn.KLDivLoss()

            # forward two model
            if conf.task == 'nlp':
                outputs_sub = sub_model(data, labels=target)
                outputs_max = max_model(data, labels=target)
                # loss = outputs[0]
            else:
                output_sub = sub_model(data)
                output_max = max_model(data)

            # KD defined loss
            loss = CE(output_max, target) + KL(output_sub, outputs_max)

            # ======== collect training feedback for other decision components [e.g., oort selector] ======

            if conf.task == 'nlp' or (conf.task == 'text_clf' and conf.model == 'albert-base-v2'):
                loss_list = [loss.item()]  # [loss.mean().data.item()]
            else:
                loss_list = loss.tolist()
                loss = loss.mean()

            temp_loss = sum(loss_list) / float(len(loss_list))
            self.loss_squre = sum([l ** 2 for l in loss_list]
                                  ) / float(len(loss_list))
            # only measure the loss of the first epoch
            if self.completed_steps < len(client_data):
                if self.epoch_train_loss == 1e-4:
                    self.epoch_train_loss = temp_loss
                else:
                    self.epoch_train_loss = (
                                                    1. - conf.loss_decay) * self.epoch_train_loss + conf.loss_decay * temp_loss

            # ========= Define the backward loss ==============
            max_optimizer.zero_grad()
            loss.backward()
            max_optimizer.step()

            # ========= Weight handler ========================
            self.optimizer.update_client_weight(
                conf, sub_model, self.global_model if self.global_model is not None else None)

            # update the weights of max_model
            max_model = load_sub_model(max_model, sub_model)

            self.completed_steps += 1

            if self.completed_steps == conf.local_steps:
                break

    def train_step(self, client_data, conf, max_model):
        for data_pair in client_data:
            # data&target preprocess
            if conf.task == 'nlp':
                (data, _) = data_pair
                data, target = mask_tokens(
                    data, tokenizer, conf, device=conf.device)
            elif conf.task == 'voice':
                (data, target, input_percentages,
                 target_sizes), _ = data_pair
                input_sizes = input_percentages.mul_(
                    int(data.size(3))).int()
            elif conf.task == 'detection':
                temp_data = data_pair
                target = temp_data[4]
                data = temp_data[0:4]
            else:
                (data, target) = data_pair
            # data preprocess
            if conf.task == "detection":
                self.im_data.resize_(data[0].size()).copy_(data[0])
                self.im_info.resize_(data[1].size()).copy_(data[1])
                self.gt_boxes.resize_(data[2].size()).copy_(data[2])
                self.num_boxes.resize_(data[3].size()).copy_(data[3])
            elif conf.task == 'speech':
                data = torch.unsqueeze(data, 1).to(device=conf.device)
            elif conf.task == 'text_clf' and conf.model == 'albert-base-v2':
                (data, masks) = data
                data, masks = Variable(data).to(
                    device=conf.device), Variable(masks).to(device=conf.device)

            else:
                data = Variable(data).to(device=conf.device)

            target = Variable(target).to(device=conf.device)

            # sample a submodel
            valid_p = [p_value for p_value in conf.p_pool if p_value <= conf.p_max]
            p_value = random.choice(valid_p)
            ratio = p_value / conf.p_max
            sub_model = sample_subnetwork(max_model, ratio)

            # set up sub_model
            sub_model.to(device=conf.device)
            sub_model.train()
            optimizer = self.get_optimizer(sub_model, conf)
            criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=conf.device)

            # forward model
            if conf.task == 'nlp':
                outputs = sub_model(data, labels=target)
                loss = outputs[0]
            elif conf.task == 'text_clf' and conf.model == 'albert-base-v2':
                outputs = sub_model(
                    data, attention_mask=masks, labels=target)
                loss = outputs.loss
                output = outputs.logits
            elif conf.task == "detection":
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = sub_model(
                    self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

                loss = rpn_loss_cls + rpn_loss_box \
                       + RCNN_loss_cls + RCNN_loss_bbox

                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls.item()
                loss_rcnn_box = RCNN_loss_bbox.item()

            else:
                output = sub_model(data)
                loss = criterion(output, target)

            # ======== collect training feedback for other decision components [e.g., oort selector] ======

            if conf.task == 'nlp' or (conf.task == 'text_clf' and conf.model == 'albert-base-v2'):
                loss_list = [loss.item()]  # [loss.mean().data.item()]

            elif conf.task == "detection":
                loss_list = [loss.tolist()]
                loss = loss.mean()
            else:
                loss_list = loss.tolist()
                loss = loss.mean()

            temp_loss = sum(loss_list) / float(len(loss_list))
            self.loss_squre = sum([l ** 2 for l in loss_list]
                                  ) / float(len(loss_list))
            # only measure the loss of the first epoch
            if self.completed_steps < len(client_data):
                if self.epoch_train_loss == 1e-4:
                    self.epoch_train_loss = temp_loss
                else:
                    self.epoch_train_loss = (
                                                    1. - conf.loss_decay) * self.epoch_train_loss + conf.loss_decay * temp_loss

            # ========= Define the backward loss ==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ========= Weight handler ========================
            self.optimizer.update_client_weight(
                conf, sub_model, self.global_model if self.global_model is not None else None)

            # update the weights of max_model
            max_model = load_sub_model(max_model, sub_model)

            self.completed_steps += 1

            if self.completed_steps == conf.local_steps:
                break


    def train(self, client_data, max_model, conf):

        clientId = conf.clientId
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        tokenizer, device = conf.tokenizer, conf.device

        max_model = max_model.to(device=device)
        max_model.train()

        trained_unique_samples = min(
            len(client_data.dataset), conf.local_steps * conf.batch_size)
        self.global_model = None

        # disable fed-prox

        # define criterion and optimizer in the train step
        error_type = None

        # NOTE: If one may hope to run fixed number of epochs, instead of iterations,
        # train for one epoch
        while self.completed_steps < conf.local_steps * len(client_data):
            try:
                if conf.kd:
                    self.train_step_kd(client_data, conf, max_model)
                else:
                    self.train_step(client_data, conf, max_model)
            except Exception as ex:
                error_type = ex
                break

        state_dicts = max_model.state_dict()
        model_param = {p: state_dicts[p].data.cpu().numpy()
                       for p in state_dicts}
        results = {'clientId': clientId, 'moving_loss': self.epoch_train_loss,
                   'trained_size': self.completed_steps * conf.batch_size,
                   'success': self.completed_steps == conf.local_steps}

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['utility'] = math.sqrt(
            self.loss_squre) * float(trained_unique_samples)
        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results


