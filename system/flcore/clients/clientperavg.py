import numpy as np
import torch
import time
import copy
import torch.nn as nn
from typing import Tuple, Union
from collections import OrderedDict
from flcore.optimizers.fedoptimizer import PerAvgOptimizer
from flcore.clients.clientbase import Client


class clientPerAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.beta = args.beta
        self.beta = self.learning_rate
        self.alpha =  self.learning_rate

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        trainloader = self.load_train_data(self.batch_size*3)
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):  # local update
            for X, Y in trainloader:
                # temp_model = copy.deepcopy(list(self.model.parameters()))
                temp_model = copy.deepcopy(self.model)

                # step 1，计算梯度并更新参数
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][:self.batch_size].to(self.device)
                    x[1] = X[1][:self.batch_size]
                else:
                    x = X[:self.batch_size].to(self.device)
                y = Y[:self.batch_size].to(self.device)
                data_batch_1 = (x, y)
                grads = self.compute_grad(temp_model, data_batch_1)
                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(self.alpha * grad)
                # step 2,只计算梯度，没有更新参数
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][self.batch_size:self.batch_size*2].to(self.device)
                    x[1] = X[1][self.batch_size:self.batch_size*2]
                else:
                    x = X[self.batch_size:self.batch_size*2].to(self.device)
                y = Y[self.batch_size:self.batch_size*2].to(self.device)
                data_batch_2 = (x, y)
                grads_1st = self.compute_grad(temp_model, data_batch_2)

                # step 3,计算 hessian矩阵，得到二阶偏倒数

                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][self.batch_size*2:].to(self.device)
                    x[1] = X[1][self.batch_size*2:]
                else:
                    x = X[self.batch_size*2:].to(self.device)
                y = Y[self.batch_size*2:].to(self.device)
                data_batch_3 = (x, y)
                grads_2nd = self.compute_grad(self.model, data_batch_3, v=grads_1st, second_order_grads=True)

                # step 4,公式有问题
                for param, grad1, grad2 in zip(self.model.parameters(), grads_1st, grads_2nd):
                    # param.data.sub_(self.beta * grad1 - self.beta * self.alpha * grad2)
                    param.data.sub_(self.beta * grad1 - self.beta * self.alpha * grad2 * grad1)



                # # restore the model parameters to the one before first update
                # for old_param, new_param in zip(self.model.parameters(), temp_model):
                #     old_param.data = new_param.data.clone()
                #
                # self.optimizer.step(beta=self.beta)

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_one_step(self, epoch):
        trainloader = self.load_train_data(self.batch_size)
        # self.model.to(self.device)
        self.model.train()

        for e in range(epoch):
            for X, Y in trainloader:
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0].to(self.device)
                    x[1] = X[1]
                else:
                    x = X.to(self.device)
                y = Y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step(beta=self.alpha)

        # self.model.cpu()


    def compute_grad(self,model: torch.nn.Module,data_batch: Tuple[torch.Tensor, torch.Tensor],v: Union[Tuple[torch.Tensor, ...], None] = None,second_order_grads=False,):
        x, y = data_batch
        if second_order_grads:
            frz_model_params = copy.deepcopy(model.state_dict())
            delta = 1e-3
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for (layer_name, param), grad in zip(model.named_parameters(), v):
                    dummy_model_params_1.update({layer_name: param + delta * grad})
                    dummy_model_params_2.update({layer_name: param - delta * grad})

            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = model(x)
            # loss_1 = self.criterion(logit_1, y) / y.size(-1)
            loss_1 = self.loss(logit_1, y)
            grads_1 = torch.autograd.grad(loss_1, model.parameters())

            model.load_state_dict(dummy_model_params_2, strict=False)
            logit_2 = model(x)
            loss_2 = self.loss(logit_2, y)
            # loss_2 = self.criterion(logit_2, y) / y.size(-1)
            grads_2 = torch.autograd.grad(loss_2, model.parameters())

            model.load_state_dict(frz_model_params)

            grads = []
            with torch.no_grad():
                for g1, g2 in zip(grads_1, grads_2):
                    grads.append((g1 - g2) / (2 * delta))
            return grads

        else:
            logit = model(x)
            # loss = self.criterion(logit, y) / y.size(-1)
            loss = self.loss(logit, y)
            grads = torch.autograd.grad(loss, model.parameters())
            return grads
