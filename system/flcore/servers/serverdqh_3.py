import copy
import os
import random

import h5py
import torch
from flcore.clients.clientdqh_3_1 import clientDQH_3_1
from flcore.clients.clientdqh_3_2 import clientDQH_3_2
from flcore.clients.clientdqh_3_3 import clientDQH_3_3
from flcore.servers.serverbase import Server
from kornia import augmentation
from torch import nn
from tqdm import tqdm
from utils.data_utils import read_client_data
from threading import Thread
import time
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
import pickle
import joblib
from utils.data_utils import read_public_data_image
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
from torchvision import transforms
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from utils.DENSE import kldiv, ImagePool, KLDiv


class FedDQH_3(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        if self.hetero_model == True:
            self.model_size_prop = args.model_size_prop
        # select slow clients
        self.set_slow_clients()
        if args.algorithm == "FedDQH31":
            self.set_clients(clientDQH_3_1)
        elif args.algorithm == "FedDQH32":
            self.set_clients(clientDQH_3_2)
        else:
            self.set_clients(clientDQH_3_3)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]

        self.loss = nn.CrossEntropyLoss()
        self.decay_rate = 0.5
        self.uploaded_heads = []

        """为了弄全局模型加入的"""
        self.optimizer_S = torch.optim.SGD(self.global_model.parameters(), lr=0.1)
        self.nz = 100 if self.dataset == 'Cifar10' or 'mnist' in self.dataset else 256
        self.channels, self.width, self.height, self.n_cls = self._get_data_info()
        # self.generator = CGeneratorA(nz=self.nz, nc=self.channels, img_size=self.width, n_cls=self.n_cls).to(
        #     self.device)

        self.generator = Generator(nz=self.nz, nc=self.channels, img_size=self.width).to(self.device)

        self.optimizer_G = torch.optim.SGD(self.generator.parameters(), lr=self.learning_rate)
        self.cls_criterion = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.diversity_criterion = DiversityLoss(metric='l1').to(self.device)
        self.aut_criterion = nn.MSELoss(reduction='mean').to(self.device)

        self.fake_data = []
        self.weight = []
        self.decay_rate = 0.5
        self.iterations = 10

        self.global_test_data = []

        self.global_model_performance = []

        self.img_size = (3, 32, 32)

        self.aug = MultiTransform([
            # global view
            transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),  # 随机裁剪
                augmentation.RandomHorizontalFlip(),  # 随机水平翻转
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                # 随机调整大小裁剪
                augmentation.RandomHorizontalFlip(),
            ]),
        ])

        if not ("Cifar" in args.dataset):
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])

        # """加入public data训练试试"""
        # self.test_data = read_public_data_image(args.dataset, 10000)
        # self.public_data_name = args.public_data
        # self.public_data_size = args.public_data_size
        # self.public_data = read_public_data_image(self.public_data_name, self.public_data_size)
        #
        # self.testloaderfull = DataLoader(self.test_data, self.batch_size, drop_last=True, shuffle=True)
        # self.trainloaderfull = DataLoader(self.public_data, self.batch_size, drop_last=True, shuffle=True)

    def train(self):
        path = 'model/FedDQH_3_client_5/alpha_01_proto/'
        # path = 'model/FedDQH_3_client_5/alpha_01/'
        for i in range(self.global_rounds + 1):
        # for i in range(1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # if self.global_protos != [None for _ in range(self.num_classes)]:
            self.send_models()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate personalized models")
            self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.global_protos = self.receive_protos()

            self.send_protos()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # for client in self.selected_clients:
        #     name = client.id
        #     file_name = path + f"{name}.pkl"
        #     # 保存数据到文件
        #     with open(file_name, 'wb') as f:
        #         pickle.dump((client.id, client.train_samples, client.sample_per_class, client.model, client.protos), f)



        # """直接读取需要的东西"""
        # s_t = time.time()
        # self.selected_clients = self.select_clients()
        # # if self.global_protos != [None for _ in range(self.num_classes)]:
        # self.send_models()
        #
        # for client in self.selected_clients:
        #     name = client.id
        #     file_name = path + f"{name}.pkl"
        #     with open(file_name, 'rb') as f:
        #         loaded_ID, loaded_train_samples, loaded_sample_per_class, loaded_model, loaded_prototype = pickle.load(f)
        #         if loaded_ID != client.id:
        #             print("文件读取出现了错误！！")
        #         client.train_samples = loaded_train_samples
        #         client.sample_per_class = loaded_sample_per_class
        #         client.model = loaded_model
        #         client.protos = loaded_prototype
        #
        # self.class_weight = np.zeros((self.num_classes, len(self.selected_clients)))
        # for i in range(self.num_classes):
        #     for c,client in enumerate(self.selected_clients):
        #         self.class_weight[i][c] = client.sample_per_class[i]
        #
        # self.class_weight = self.class_weight / self.class_weight.sum(axis=1, keepdims=True)
        #
        # self.receive_models()
        # self.global_protos = self.receive_protos()
        #
        # for client in self.selected_clients:
        #     client_test_data = read_client_data(client.dataset, client.id, is_train=False)
        #     self.global_test_data.extend(client_test_data)
        #
        #
        # self.ensemble_model = Ensemble_model(self.selected_clients, self.class_weight)
        #
        # self.save_dir = './fig'
        # self.data_pool = ImagePool(root=self.save_dir)
        #
        # """测试模型"""
        # self.test_client_model()
        # self.test_ensemble_model()
        #
        # """固定head"""
        # weight = []
        # for [label, proto_list] in self.global_protos.items():
        #     l = 0
        #     proto_onehot = np.zeros(self.num_classes)
        #     proto_onehot[label] = 1
        #     proto_onehot = torch.Tensor(proto_onehot)
        #     for client in self.selected_clients:
        #         out = client.model.head(proto_list).to("cpu")
        #         l += self.loss(out, proto_onehot)
        #     weight.append(np.exp(-self.decay_rate * l))
        # weights = [wei / sum(weight) for wei in weight]
        #
        # for param in self.global_model.head.parameters():
        #     param.data.zero_()
        #
        # for wei, client_head in zip(weights, self.uploaded_heads):
        #     for head_param, client_param in zip(self.global_model.head.parameters(), client_head.parameters()):
        #         head_param.data += client_param.data.clone() * wei
        #
        #
        #
        # """开始训练全局模型"""
        # label_start = 0
        # for i in range(1000):
        #     print(f"\n-------------Round number: {i}-------------")
        #     print("\nEvaluate global model")
        #     self.global_evaluate()
        #
        #     """为了训练全局模型加入的代码"""
        #     # labels_all = self.generate_labels(self.batch_size * self.iterations)
        #     best_cost = 1e6
        #     best_inputs = None
        #     lambda_cls = 1
        #     lambda_dis = 1
        #     labels = np.array([(label_start + i) % self.num_classes for i in range(self.batch_size)])
        #     label_start += self.batch_size
        #     # labels = labels_all[e * self.batch_size:(e * self.batch_size + self.batch_size)]
        #     batch_weight = torch.Tensor(self.get_batch_weight(labels)).to(self.device)
        #     onehot = np.zeros((self.batch_size, self.num_classes))  # shape=(10, 10)
        #     onehot[np.arange(self.batch_size), labels] = 1
        #     y_onehot = torch.Tensor(onehot).to(self.device)
        #     # z = torch.randn((self.batch_size, self.nz, 1, 1)).to(self.device)
        #
        #     z = torch.randn(size=(self.batch_size, self.nz)).cuda()  # 噪音
        #     z.requires_gard = True
        #
        #     for e in range(self.iterations):
        #         ############## train generator ##############
        #         self.global_model.eval()
        #         self.generator.train()
        #         loss_G = 0
        #         loss_md_total = 0
        #         loss_aut_total = 0
        #         loss_ap_total = 0
        #
        #         loss_aut_list = []
        #         loss_md_list = []
        #         loss_ap_list = []
        #
        #         y = torch.Tensor(labels).long().to(self.device)
        #
        #
        #          # 但是，这里有个bug，其实我们在喂给每个clientmodel的图像都不一样的，每个client都会重新生成fake data，这是可以的吗？按照直觉来说，应该是拿同一个sample来喂给模型更合理，但是，好像它这里给每个client都喂不同的模型也有道理，这样还能更多的训练模型，但是好像也只是多样性来说更有保证而已
        #         self.optimizer_G.zero_grad()
        #         # fake = self.generator(z, y_onehot)
        #         fake = self.generator(z)
        #         fake, _ = self.aug(fake)  # crop and normalize  对生成的数据进行增强
        #         g_protos = []
        #         for yy in y:
        #             g_protos.append(self.global_protos.get(int(yy)))
        #
        #         g_protos = torch.stack(g_protos)
        #         s_logit = self.global_model(fake)
        #         t_protos = self.ensemble_model.get_feature(fake, y)
        #         t_logit = self.ensemble_model(fake, y)
        #
        #         """
        #         这里考虑是否要平均全部非零的结果。我认为应该是要的。因为非零的结果根本就没有起到任何作用。
        #         如果，一个非常重要的指标，为【0,100,0，0】平均下来就只有25了，而这会阻碍模型的训练。
        #         """
        #
        #         # # 计算欧氏距离
        #         # euclidean_distance = torch.norm(t_prototypes - c_protos, dim=1)
        #         # weighted_euclidean_distance = euclidean_distance * batch_weight[:, c]
        #
        #         # 计算曼哈顿距离
        #         manhattan_distance = torch.sum(torch.abs(t_protos - g_protos), dim=1)
        #         loss_aut = torch.mean(manhattan_distance)
        #
        #         loss_md = - torch.mean(
        #             torch.mean(torch.abs(s_logit - t_logit.detach()), dim=1))
        #
        #         loss_ap = self.diversity_criterion(z.view(z.shape[0], -1), fake)
        #         mask = (s_logit.max(1)[1] != t_logit.max(1)[1]).float()
        #         loss_adv = -(kldiv(s_logit, t_logit, reduction='none').sum(1)*mask).mean()
        #         loss_oh = F.cross_entropy(t_logit, y)  # ce_loss
        #         # loss = loss_md + lambda_cls * loss_aut + lambda_dis * loss_ap+loss_adv+loss_oh
        #         loss = loss_adv + loss_oh + loss_aut
        #
        #         if best_cost > loss.item() or best_inputs is None:
        #             best_cost = loss.item()
        #             best_inputs = fake.data
        #
        #
        #         # vutils.save_image(best_inputs.clone(), '1.png', normalize=True, scale_each=True, nrow=10)
        #
        #         loss.backward()
        #         self.optimizer_G.step()
        #
        #         # loss.backward()
        #         loss_G += loss
        #         loss_md_total += loss_md
        #         loss_aut_total += loss_aut
        #         loss_ap_total += loss_ap
        #
        #         loss_aut_list.append(loss_aut)
        #         loss_md_list.append(loss_md)
        #         loss_ap_list.append(loss_ap)
        #         # save best inputs and reset data iter
        #
        #     self.data_pool.add(best_inputs, y)  # 生成了一个batch的数据
        #     print(self.data_pool._idx)
        #
        #     ############## train student model##############
        #     self.global_model.train()
        #     self.generator.eval()
        #
        #     criterion = KLDiv(1)  # （T表示温度）
        #     total_loss = 0.0
        #     correct = 0.0
        #     datasets = self.data_pool.get_dataset(transform=self.transform)  # 获取程序运行到现在所有的图片
        #     data_loader = torch.utils.data.DataLoader(
        #         datasets, batch_size=self.batch_size, shuffle=True,
        #         num_workers=4, pin_memory=True, )
        #     with tqdm(data_loader, ncols=85, disable=True) as epochs:  # 读取之前生成的数据
        #         for idx, (images) in enumerate(epochs):
        #             self.optimizer_S.zero_grad()
        #             images = images.cuda()
        #             with torch.no_grad():
        #                 t_out = self.ensemble_model(images)
        #             s_out = self.global_model(images.detach())
        #             loss_s = criterion(s_out, t_out.detach())
        #
        #             loss_s.backward()
        #             self.optimizer_S.step()
        #
        #             total_loss += loss_s.detach().item()
        #             pred = s_out.argmax(dim=1)
        #             target = t_out.argmax(dim=1)
        #             correct += pred.eq(target.view_as(pred)).sum().item()
        #
        #         # for param in self.global_model.base.parameters():
        #         #     param.requires_grad = True
        #         # for param in self.global_model.head.parameters():
        #         #     param.requires_grad = False
        #         #
        #         # fake_data = []
        #         # weight_cri = []
        #         #
        #         # self.fake_data = fake_data
        #         # self.weight = weight_cri
        #         #
        #         # for _ in range(5):
        #         #     self.optimizer_S.zero_grad()
        #         #     fake = self.generator(z, y_onehot).detach()
        #         #     s_logit = self.global_model(fake)
        #         #
        #         #     y = torch.Tensor(labels).long().to(self.device)
        #         #     g_protos = []
        #         #     for yy in y:
        #         #         g_protos.append(self.global_protos.get(int(yy)))
        #         #     g_protos = torch.stack(g_protos)
        #         #     s_prototype = self.global_model.base(fake)
        #         #
        #         #     loss_aut = torch.sum(torch.abs(s_prototype - g_protos))
        #         #     loss_s = self.cls_criterion(s_logit, y_onehot)
        #         #
        #         #
        #         #     t_logit = self.ensemble_model(fake)
        #         #     loss_logit = torch.mean(-F.log_softmax(s_logit, dim=1) * F.softmax(t_logit, dim=1))
        #         #
        #         #     loss = loss_logit + loss_aut + loss_s
        #         #
        #         #     # """loss global"""
        #         #     # loss = loss_aut + loss_s
        #         #
        #         #     # """loss client"""
        #         #     # for c, client in enumerate(self.active_clients):
        #         #     #     t_prototype = client.model.base(fake).detach()
        #         #     #     t_logit = client.model(fake).detach()
        #         #     #
        #         #     #     loss_proto_all = self.aut_criterion(s_prototype.detach(), t_prototype) * batch_weight[:, c]
        #         #     #
        #         #     #     # 使用索引获取非零元素
        #         #     #     loss_proto_non_zeros = torch.nonzero(loss_proto_all)
        #         #     #
        #         #     #     loss_proto = torch.mean(loss_proto_all[loss_proto_non_zeros].float())
        #         #     #
        #         #     #     loss_logit += torch.mean(-F.log_softmax(s_logit, dim=1) * F.softmax(t_logit, dim=1))
        #         #     #     # loss_logit += F.kl_div(s_logit.log(), t_logit, reduction='batchmean')*batch_weight[:,c]
        #         #     # loss = loss_logit + loss_proto
        #         #
        #         #     # self.optimizer_S.zero_grad()
        #         #     loss.backward()
        #         #     # print("global model的损失为：%d", loss)
        #         #     self.optimizer_S.step()

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        if self.rs_test_acc != []:
            print(max(self.rs_test_acc))
            self.save_results()
        # print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        """
        这些是要加回来的
        """
        # self.test_client_model()
        # self.test_ensemble_model()
        # self.save_global_performance()

    def save_global_performance(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/global_model_performance/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.global_model_performance)):
            algo1 = algo + "_global_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo1)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('test_global', data=self.global_model_performance)

            file_name = result_path + f"{algo1}.pkl"
            # 保存数据到文件
            with open(file_name, 'wb') as f:
                pickle.dump((self.global_model), f)

            best_acc = max(self.global_model_performance)
            print(f"Best accuracy:{best_acc}, round:{self.global_model_performance.index(best_acc)}")


    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_class_weight = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
            self.uploaded_class_weight.append(client.sample_per_class)

        uploaded_class_weight = defaultdict(list)
        agg_protos_label = defaultdict(list)

        for index, local_protos in enumerate(self.uploaded_protos):  # 每个client的proto
            for label in local_protos.keys():
                agg_protos_label[label].append(local_protos[label])
                uploaded_class_weight[label].append(self.uploaded_class_weight[index][label]/sum(row[label] for row in self.uploaded_class_weight))

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for index, i in enumerate(proto_list):
                    proto += i.data*uploaded_class_weight[label][index]  # 需要权重
                agg_protos_label[label] = proto
            else:
                agg_protos_label[label] = proto_list[0].data

        return agg_protos_label



    def evaluate(self, acc=None, loss=None, printting=True):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        if printting:
            print("Averaged Train Loss: {:.4f}".format(train_loss))
            print("Averaged Test Accurancy: {:.4f}".format(test_acc))
            # self.print_(test_acc, train_acc, train_loss)
            print("Std Test Accurancy: {:.4f}".format(np.std(accs)))

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_heads = []


        tot_samples = 0
        uploaded_weights = []
        self.uploaded_samples = []
        self.uploaded_client_class_weight = []

        for client in self.active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                uploaded_weights.append(client.sample_per_class)

                self.uploaded_heads.append(client.model.head)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        for c in range(self.num_classes):
            weight = [tensor[c] for tensor in uploaded_weights]
            self.uploaded_samples.append(np.sum(weight))
            weight = weight / np.sum(weight)
            self.uploaded_client_class_weight.append(weight)  # 我这里是按每一类的client来分的



    def send_models(self):
        assert (len(self.clients) > 0)
        if self.uploaded_heads != []:
            for client in self.selected_clients:
                start_time = time.time()
                weight = []
                for c in self.selected_clients:
                    w = 0
                    for [label, proto_list] in client.protos.items():
                        out = (c.model.head(proto_list)).to("cpu")
                        ww = client.sample_per_class[label] / client.train_samples
                        onehot = np.zeros(self.num_classes)
                        onehot[label] = 1
                        y_onehot = torch.Tensor(onehot)
                        l = self.loss(out, y_onehot).detach()
                        w += l * ww
                    weight.append(np.exp(-self.decay_rate * w))

                weights = [wei / sum(weight) for wei in weight]

                head = copy.deepcopy(self.uploaded_heads[0])
                for param in head.parameters():
                    param.data.zero_()

                for wei, client_head in zip(weights, self.uploaded_heads):
                    for head_param, client_param in zip(head.parameters(), client_head.parameters()):
                        head_param.data += client_param.data.clone() * wei
                client.set_parameters(head)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

            # # 如果没有被选择，就直接退化为传输global model
            # for client in self.clients:
            #     if client not in self.selected_clients:
            #         start_time = time.time()
            #
            #         client.set_parameters(self.global_model)
            #
            #         client.send_time_cost['num_rounds'] += 1
            #         client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
        # else:
        #     for client in self.clients:
        #         start_time = time.time()
        #
        #         client.set_parameters(self.global_model)
        #
        #         client.send_time_cost['num_rounds'] += 1
        #         client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    """
    为了弄全局模型加入的
    """
    def generate_labels(self, number):
        cls_num = np.array(self.uploaded_samples)
        labels = np.arange(number)
        proportions = cls_num / cls_num.sum()
        proportions = (np.cumsum(proportions) * number).astype(int)[:-1]
        labels_split = np.split(labels, proportions)
        for i in range(len(labels_split)):
            labels_split[i].fill(i)
        labels = np.concatenate(labels_split)
        np.random.shuffle(labels)
        return labels.astype(int)

    def get_batch_weight(self, labels):
        bs = labels.size
        cls_clnt_weight = np.array(self.uploaded_client_class_weight)
        num_clients = cls_clnt_weight.shape[1]
        batch_weight = np.zeros((bs, num_clients))
        batch_weight[np.arange(bs), :] = cls_clnt_weight[labels, :]
        return batch_weight

    def _get_data_info(self):
        if "Cifar100" in self.dataset:
            return [3, 32, 32, 100]
        elif "mnist" in self.dataset:
            return [1, 28, 28, 10]
        elif 'Tiny-imagenet' in self.dataset:
            return [3, 64, 64, 200]
        elif "Cifar10" in self.dataset:
            return [3, 32, 32, 10]
        else:
            raise ValueError('Wrong dataset.')

    def send_global_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_global_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def global_evaluate(self):
        globaltestloadar = DataLoader(self.global_test_data, self.batch_size, drop_last=True, shuffle=True)

        test_acc = 0
        test_num = 0

        with torch.no_grad():
            for x, y in globaltestloadar:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        acc_global = test_acc / test_num

        if acc_global is not None:
            self.global_model_performance.append(acc_global)

        print(f"global model的全局的ACC为{acc_global}")

    def test_global_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_global_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_global_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_global_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def set_clients(self, clientObj):
        if self.hetero_model == True:
            num_ones = int(self.num_clients * self.model_size_prop)
            model_sizes = [1] * num_ones + [0] * (self.num_clients - num_ones)
            random.shuffle(model_sizes)
            for i, train_slow, send_slow, model_size in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients, model_sizes):
                train_data = read_client_data(self.dataset, i, is_train=True)
                test_data = read_client_data(self.dataset, i, is_train=False)

                client = clientObj(self.args,
                                id=i,
                                train_samples=len(train_data),
                                test_samples=len(test_data),
                                train_slow=train_slow,
                                send_slow=send_slow,
                                model_size=model_size)
                self.clients.append(client)
        else:
            for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients,
                                                self.send_slow_clients):
                train_data = read_client_data(self.dataset, i, is_train=True)
                test_data = read_client_data(self.dataset, i, is_train=False)
                client = clientObj(self.args,
                                   id=i,
                                   train_samples=len(train_data),
                                   test_samples=len(test_data),
                                   train_slow=train_slow,
                                   send_slow=send_slow)
                self.clients.append(client)

    def test_client_model(self):
        globaltestloadar = DataLoader(self.global_test_data, self.batch_size, drop_last=False, shuffle=True)

        for client in self.selected_clients:
            test_acc = 0
            test_num = 0

            with torch.no_grad():
                for x, y in globaltestloadar:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client.model(x)

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]


            acc_global = test_acc/test_num
            acc_local = client.test_model(client.model)

            print(f"client {client.id}-th的本地ACC为{acc_local}, 全局的ACC为{acc_global}")



    def test_ensemble_model(self):
        globaltestloadar = DataLoader(self.global_test_data, self.batch_size, drop_last=False, shuffle=True)

        test_acc = 0
        test_num = 0

        with torch.no_grad():
            for x, y in globaltestloadar:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                ensemble_output = self.ensemble_model(x, y)

                test_acc += (torch.sum(torch.argmax(ensemble_output.cuda(), dim=1) == y)).item()
                test_num += y.shape[0]

        acc_global = test_acc/test_num

        # auc_local = client.test_model(client.model)
        acc_local = {}
        for client in self.selected_clients:
            acc_local[client.id] = client.test_model(self.ensemble_model)

        print(f"ensemble_model的的本地的ACC为{acc_local}, 全局的ACC为{acc_global}")



class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img



class CGeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, n_cls=10):
        super(CGeneratorA, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * self.init_size ** 2))  # 100， 4096
        self.l2 = nn.Sequential(nn.Linear(n_cls, ngf * self.init_size ** 2))  # 10， 4096

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),  # 128
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),  # 128, 128, 3, 1, 1
            nn.BatchNorm2d(ngf * 2),  # 128
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),  # 128 , 64, 3, 1, 1
            nn.BatchNorm2d(ngf),  # 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),  # 64, 1, 3, 3, 1, 1
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z, y):
        out_1 = self.l1(z.view(z.shape[0], -1))  # Tensor(10, 4096)
        out_2 = self.l2(y.view(y.shape[0], -1))  # Tensor(10, 4096)
        out = torch.cat([out_1, out_2], dim=1)  # Tensor(10, 8192)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)  # Tensor(10, 128, 8, 8)
        img = self.conv_blocks0(out)  # Tensor(10, 128, 8, 8)
        img = nn.functional.interpolate(img, scale_factor=2)  # Tensor(10, 128, 16, 16)
        img = self.conv_blocks1(img)  # Tensor(10, 128, 16, 16)
        img = nn.functional.interpolate(img, scale_factor=2)  # Tensor(10, 128, 32, 32)
        img = self.conv_blocks2(img)  # Tensor(10, 3, 32, 32)
        return img

class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out

class Ensemble_model(torch.nn.Module):
    def __init__(self, clients, wei):
        super(Ensemble_model, self).__init__()
        self.num = len(clients)
        self.models = []
        self.wei = torch.tensor(wei).cuda()
        for client in clients:
            self.models.append(client.model)

    def forward(self, x, y=None):
        logits_e = 0
        if y != None:
            wei_batch = torch.zeros(len(y), self.num).cuda()
            for i, yy in enumerate(y):
                wei_batch[i] = self.wei[int(yy)]

            for i in range(self.num):
                logit = self.models[i](x)
                logits_e += logit * wei_batch[:,i].unsqueeze(1)
        else:
            for i in range(self.num):
                logits = self.models[i](x)
                logits_e += logits
            logits_e = logits_e / self.num

        return logits_e

    def get_feature(self, x, y):
        feature_total = 0
        wei_batch = torch.zeros(len(y), self.num).cuda()
        for i, yy in enumerate(y):
            wei_batch[i] = self.wei[int(yy)]
        for i in range(self.num):
            feature = self.models[i].base(x)
            feature_total += feature * wei_batch[:,i].unsqueeze(1)

        return feature_total



class MultiTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)