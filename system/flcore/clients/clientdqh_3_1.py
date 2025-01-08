from collections import defaultdict
import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.trainmodel.models import *
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class clientDQH_3_1(Client):
    def __init__(self, args, id, train_samples, test_samples, model_size=-1, **kwargs):
        super().__init__(args, id, train_samples, test_samples, model_size, **kwargs)

        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

        self.global_model = None


        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma
        )

        self.plocal_steps = args.plocal_epochs

        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:  # 统计每个client的每个种类有多少数据，因为前面是按照数据的batchsize来读取的，所以这里的for也是按照batchsize来的，如y=tensor([7, 3, 3, 9, 4, 4, 3, 3, 5, 3])
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        print(self.id, self.sample_per_class)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        # self.collect_protos()
        self.protos = agg_func(protos)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def test_model(self, model):
        testloaderfull = self.load_test_data()
        model.eval()

        test_acc = 0
        test_num = 0

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if type(model) == type(self.model):
                    output = model(x)
                else:
                    output = model(x,y)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        acc = test_acc/test_num

        return acc

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

    def set_parameters(self, model):
        layers1 = len(list(model.children()))
        layers2 = len(list(self.model.children()))
        if layers1 == layers2:
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()
        else:
            for new_param, old_param in zip(model.parameters(), self.model.head.parameters()):
                old_param.data = new_param.data.clone()

    def set_global_parameters(self, global_model):
        if self.global_model == None:
            self.global_model = copy.deepcopy(global_model)
        else:
            # pre_global_model = copy.deepcopy(self.global_model.state_dict())

            for new_param, old_param in zip(global_model.parameters(), self.global_model.parameters()):
                old_param.data = new_param.data.clone()

            # current_global_model_state = self.global_model.state_dict()
            #
            # model_changed = False
            # for key in pre_global_model.keys():
            #     if not torch.all(torch.eq(pre_global_model[key], current_global_model_state[key])):
            #         model_changed = True
            #         break
            #
            # if model_changed:
            #     print("模型发生了改变")
            # else:
            #     print("模型未发生改变")


    def test_global_metrics(self):
        testloaderfull = self.load_test_data()
        self.global_model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.global_model(x)

                # if self.dp:
                #     output = self.pred(output.detach())

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_global_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.global_model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos