import copy

import torch
import os
import torch.multiprocessing as mp
from tqdm import tqdm
import torch.nn as nn
from FLAlgorithms.users.userFedMEKTC import userMultimodalRepFusion
from FLAlgorithms.users.userbase import User
# from FLAlgorithms.servers.serverbase_dem import Dem_Server
from FLAlgorithms.servers.serverbase import Server
from Setting import rs_file_path, N_clients
from utils.data_utils import write_file
from utils.dem_plot import plot_from_file,plot_from_file2
from utils.model_utils import make_seq_batch,get_seg_len,load_data,client_idxs, split_public,get_seg_len_public
from torch.utils.data import DataLoader
import numpy as np
# from FLAlgorithms.optimizers.fedoptimizer import DemProx_SGD
# from FLAlgorithms.trainmodel.models import *
# Implementation for Server
import json, codecs

from utils.train_utils import KL_Loss, JSD, DCCLoss
from Setting import *
from torch import nn, optim
from sklearn.metrics import f1_score
import sys
class MultimodalRepFusion(Server):
    def __init__(self, train_A, train_B,experiment, device, dataset, algorithm, model,  model_server,embedding_layer,embedding_layer1, batch_size,
                 learning_rate, num_glob_iters, local_epochs, optimizer, num_users, times , cutoff,args):
        super().__init__(train_A, train_B,experiment, device, dataset, algorithm, model[0],  model_server[0],embedding_layer[0],embedding_layer1[0],batch_size,
                         learning_rate, num_glob_iters,local_epochs, optimizer, num_users, times,args)

        # Initialize data for all  users
        if DATASET == "ur_fall":
            self.batch_min = 16
            self.batch_max = 32
        else:
            self.batch_min = 128
            self.batch_max = 256
        self.train_A = train_A
        self.train_B = train_B
        self.eval_interval = 1
        self.loss = nn.CrossEntropyLoss()

        # self.optimizer = DemProx_SGD(self.model.parameters(), lr=global_learning_rate, mu=0)

        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=global_learning_rate)
        self.criterion_MSE = nn.MSELoss().to(self.device)
        self.rep_size = rep_size
        self.criterion_DCC = DCCLoss(self.rep_size, self.device)
        self.optimizer = optim.Adam(self.model_server.parameters(), lr=global_learning_rate)
        self.optimizer_glob_ae = optim.Adam(self.model.parameters(), lr=global_ae_learning_rate)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion_KL = KL_Loss(temperature=3.0)
        self.criterion_JSD = JSD()
        self.cos = nn.CosineSimilarity()
        self.test_modality = test_modality
        self.avg_local_dict_prev_1 = dict()
        self.gamma = gamma
        self.model_ae = MODEL_AE
        self.num_clients_A = NUM_CLIENT_A
        self.num_clients_B = NUM_CLIENT_B
        self.num_clients_AB = NUM_CLIENT_AB

        # total_users = len(dataset[0][0])
        self.sub_data = cutoff
        if(self.sub_data):
            randomList = self.get_partion(self.total_users)
        #

        # self.publicdatasetlist= DataLoader(public_data, self.batch_size, shuffle=False)  # no shuffle
        sample=[]
        testing_sample=[]

        server_test = load_data(dataset)[1]

        public = load_data(dataset)[2]
        # print('public len', len(public["y"]))
        self.public = public
        # print('public len', len(self.public["y"]))
        self.data_test = server_test


        testing_sample = []
        modalities = ["A" for _ in range(self.num_clients_A)] + ["B" for _ in range(
            self.num_clients_B)] + ["AB" for _ in range(self.num_clients_AB)]
        for i in range(self.total_users):
            client_train = load_data(dataset)[0]
            id = client_idxs(client_train)
            public_data = load_data(dataset)[2]

            # if(self.sub_data):
            #     if(i in randomList):
            #         train, test = self.get_data(train, test)
            user = userMultimodalRepFusion(device, id[i], client_train, public_data, model, model_server , embedding_layer,embedding_layer1,modalities[i], batch_size, learning_rate, beta,
                            local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        self.seg_len_public = get_seg_len_public(len(self.public["A"]))
        # print("public len",len(public))
            # print(user.train_samples)
        # print("train sample median :", np.median(training_sample))
        # print("test sample median :", np.median(testing_sample))
        print("sample is", np.median(sample))




            
        print("Fraction number of users / total users:",num_users, " / " ,self.total_users)

        print("Finished creating server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)
    def freeze(self, sub_model):
        """Freeze the parameters of a model"""
        for param in sub_model.parameters():
            param.requires_grad = False

    def unfreeze(self, sub_model):
        """Unfreeze the parameters of a model"""
        for param in sub_model.parameters():
            param.requires_grad = True

    def serialize_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return np.asscalar(obj)
        elif isinstance(obj, np.datetime64):
            return obj.astype(str)
        elif np.isnan(obj):
            return 'NaN'
        elif np.isinf(obj):
            return 'Infinity' if obj >0 else '-Infinity'
        else:
            raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")


    def global_ae_distill(self, epochs,global_epochs):
        # self.freeze(self.model)

        self.model.train()


        repA_local= dict()
        repA_local_full= dict()
        repB_local = dict()
        repB_local_full = dict()
        repA_avg=dict()
        repB_avg= dict()
        repA1_local= dict()
        repA1_local_full= dict()
        repB1_local = dict()
        repB1_local_full = dict()
        repA1_avg=dict()
        repB1_avg= dict()


        a = 0
        b = 0
        k = 0
        n_A = 0
        n_B = 0
        for user in self.selected_users:
            if user.modality == "A" or user.modality == "AB":
                n_A += 1
            if user.modality == "B" or user.modality == "AB":
                n_B += 1
            k+=1

        for user in self.selected_users:
            # batch_size = np.random.randint(
            #     low=self.batch_min, high=self.batch_max)
            if user.modality == "A" or user.modality == "AB":
                a += 1
                repA_local.clear()
                repA1_local.clear()
            if user.modality == "B" or user.modality == "AB":
                b += 1
                repB1_local.clear()
                repB_local.clear()
            # repA_local.clear()
            # repB_local.clear()
            # repA1_local.clear()
            # repB1_local.clear()
            if DATASET == "ur_fall":
                batch_size = 24
            else:
                batch_size = 200

            # A_public, B_public, _ = make_seq_batch(
            #     self.public, [0], len(self.public["A"]), batch_size)
            # seq_len_public = A_public.shape[1]
            A_public, B_public, _ = make_seq_batch(
                self.public, [0], self.seg_len_public, batch_size)
            seq_len_public = A_public.shape[1]
            idx_start_public = 0
            idx_end_public = 0
            # print("public len is ",len(A_public))
            while idx_end_public < seq_len_public:
                # win_len = np.random.randint(low=16, high=32)
                win_len = 24
                idx_start_public = idx_end_public
                idx_end_public += win_len
                idx_end_public = min(idx_end_public, seq_len_public)
                if user.modality == "A" or user.modality == "AB":
                    x_A_public = A_public[:, idx_start_public:idx_end_public, :]
                    seq_A_public = torch.from_numpy(x_A_public).double().to(self.device)
                    repA_public, repA_public1 = user.model.encode(seq_A_public, "A")
                    # repA_public= user.model.encode(seq_A_public, "A")
                    repA_public = repA_public.cpu().detach().numpy()
                    repA_public1 = repA_public1.cpu().detach().numpy()
                    repA_local[idx_end_public]= repA_public
                    repA1_local[idx_end_public] = repA_public1
                if user.modality == "B" or user.modality == "AB":

                    x_B_public = B_public[:, idx_start_public:idx_end_public, :]
                    seq_B_public = torch.from_numpy(x_B_public).double().to(self.device)
                    repB_public, repB_public1 = user.model.encode(seq_B_public, "B")
                    # repB_public = user.model.encode(seq_B_public, "B")
                    repB_public = repB_public.cpu().detach().numpy()
                    repB_public1 = repB_public1.cpu().detach().numpy()
                    repB_local[idx_end_public] = repB_public
                    repB1_local[idx_end_public] = repB_public1

            if user.modality == "A" or user.modality == "AB":
                repA_local_full[a] = repA_local
                repA1_local_full[a] = repA1_local
            if user.modality == "B" or user.modality == "AB":
                repB_local_full[b] = repB_local
                repB1_local_full[b] = repB1_local

        # save_dir = "./knowledge/MM-FedMEKT-C"
        #
        # knowledge_A=dict()
        # knowledge_A1 = dict()
        # knowledge_B = dict()
        # knowledge_B1 = dict()
        # file_pathA = os.path.join(save_dir, 'knowledgeA_r_{}.json'.format(global_epochs))
        # file_pathA1 = os.path.join(save_dir, 'knowledgeA1_r_{}.json'.format(global_epochs))
        # file_pathB = os.path.join(save_dir, 'knowledgeB_r_{}.json'.format(global_epochs))
        # file_pathB1 = os.path.join(save_dir, 'knowledgeB1_r_{}.json'.format(global_epochs))


        # print("rep B full", repB_local_full)
        n=0
        for client in repA_local_full.keys():
            repA = repA_local_full[client]
            for key in repA_local_full[client].keys():
                if (n==0):
                    repA_avg[key] = repA[key]/n_A
                else:
                    repA_avg[key] +=repA[key]/n_A
                # knowledge_A[key] =repA_avg[key].tolist()
            n+=1
        # print(repA_test)


        m=0
        for client in repB_local_full.keys():
            repB = repB_local_full[client]
            for key in repB_local_full[client].keys():
                if (m == 0):
                    repB_avg[key] = repB[key] / n_B
                else:
                    repB_avg[key] += repB[key] / n_B
                # knowledge_B[key] = repB_avg[key].tolist()
            m += 1
        g = 0
        for client in repA1_local_full.keys():
            repA1 = repA1_local_full[client]
            for key in repA1_local_full[client].keys():
                if (g == 0):
                    repA1_avg[key] = repA1[key] /n_A
                else:
                    repA1_avg[key] += repA1[key] / n_A
                # knowledge_A1[key] = repA1_avg[key].tolist()
            g += 1

        j = 0
        for client in repB1_local_full.keys():
            repB1 = repB1_local_full[client]
            for key in repB1_local_full[client].keys():
                if (j == 0):
                    repB1_avg[key] = repB1[key] / n_B
                else:
                    repB1_avg[key] += repB1[key] / n_B
                # knowledge_B1[key] = repB1_avg[key].tolist()
            j += 1
        # print("local rep B is", repB_avg)

        #Calculate knowledge size
        # size_bytes_A= sys.getsizeof(repA_avg)
        # size_kb_A = size_bytes_A / 1024
        # print(f"Rep A size:{size_kb_A:.2f} KB")
        #
        # size_bytes_A1 = sys.getsizeof(repA_avg)
        # size_kb_A1= size_bytes_A1 / 1024
        # print(f"Rep A1 size:{size_kb_A1:.2f} KB")
        #
        # size_bytes_B = sys.getsizeof(repB_avg)
        # size_kb_B = size_bytes_B / 1024
        # print(f"Rep B size:{size_kb_B:.2f} KB")
        #
        # size_bytes_B1 = sys.getsizeof(repB1_avg)
        # size_kb_B1 = size_bytes_B1 / 1024
        # print(f"Rep B1 size:{size_kb_B1:.2f} KB")

        # #Save knowledge to json file
        # with open(file_pathA, 'w') as f:
        #     json.dump(knowledge_A, f)
        # with open(file_pathA1, 'w') as f:
        #     json.dump(knowledge_A1, f)
        # with open(file_pathB, 'w') as f:
        #     json.dump(knowledge_B, f)
        # with open(file_pathB1, 'w') as f:
        #     json.dump(knowledge_B1, f)


        # TODO: Avg rep local output according num of samples of each users
        # TODO: Sum of DIR regularizer :done
        #
        # if self.gamma < 0.8 and glob_iter>=30:
        #     self.gamma += (0.8-0.5)/(NUM_GLOBAL_ITERS-30)
        #     # self.gamma += (0.5 - 0.05) / NUM_GLOBAL_ITERS

        Global_Rec_round_loss = []
        Global_KT_round_loss = []

        #Global distillation
        # TODO: implement several global iterations to construct generalized knowledge :done
        for epoch in range(1, epochs+1):
            Global_Rec_loss = []
            Global_KT_Loss = []

            self.model.train()

            # for batch_idx, (X_public, y_public) in enumerate(self.publicloader):
            # batch_size = np.random.randint(
            #     low=self.batch_min, high=self.batch_max)
            if DATASET == "ur_fall":
                batch_size = 24
            else:
                batch_size = 200
            # print("batch size is", batch_size)
            # A_public, B_public, _ = make_seq_batch(
            #     self.public, [0], len(self.public["A"]), batch_size)
            # seq_len_public = A_public.shape[1]

            A_public, B_public, _ = make_seq_batch(
                self.public, [0], self.seg_len_public, batch_size)
            seq_len_public = A_public.shape[1]

            idx_start_public = 0
            idx_end_public = 0
            a += 1
            while idx_end_public < seq_len_public:
                    win_len = 24
                    # win_len = np.random.randint(low=16, high=32)
                    idx_start_public = idx_end_public
                    idx_end_public += win_len
                    idx_end_public = min(idx_end_public, seq_len_public)

                    x_A_public = A_public[:, idx_start_public:idx_end_public, :]
                    seq_A_public = torch.from_numpy(x_A_public).double().to(self.device)
                    inv_idx_public = torch.arange(seq_A_public.shape[1] - 1, -1, -1).long()
                    x_B_public = B_public[:, idx_start_public:idx_end_public, :]
                    seq_B_public = torch.from_numpy(x_B_public).double().to(self.device)
                    inv_idx_public = torch.arange(seq_B_public.shape[1] - 1, -1, -1).long()
                    self.optimizer_glob_ae.zero_grad()
                    repA_avg_local = torch.from_numpy(repA_avg[idx_end_public]).double().to(self.device)
                    repA1_avg_local = torch.from_numpy(repA1_avg[idx_end_public]).double().to(self.device)
                    repB_avg_local = torch.from_numpy(repB_avg[idx_end_public]).double().to(self.device)
                    repB1_avg_local = torch.from_numpy(repB1_avg[idx_end_public]).double().to(self.device)

                    Global_ReconstructionLoss = []
                    Global_Knowledge_Transfer_Loss = []
                    # self.freeze(self.model.encoder_B)
                    repA_public, repA_public1 = self.model.encode(seq_A_public, "A")
                    repB_public, repB_public1 = self.model.encode(seq_B_public, "B")
                    # repA_public= self.model.encode(seq_A_public, "A")
                    # repB_public= self.model.encode(seq_B_public, "B")
                    embedding_knowledge_local = torch.cat([repA_avg_local, repB_avg_local], dim=0)
                    embedding_knowledge_local=self.embedding_layer(embedding_knowledge_local)
                    embedding_knowledge_local1 = torch.cat([repA1_avg_local, repB1_avg_local], dim=0)
                    embedding_knowledge_local1 = self.embedding_layer1(embedding_knowledge_local1)
                    embedding_knowledge_global = torch.cat([repA_public, repB_public], dim=0)
                    embedding_knowledge_global = self.embedding_layer(embedding_knowledge_global)
                    embedding_knowledge_global1 = torch.cat([repA_public1, repB_public1], dim=0)
                    embedding_knowledge_global1 = self.embedding_layer1(embedding_knowledge_global1)
                    output_A, output_B = self.model(seq_A_public, "A")

                    # print("size of rep is ", repA_public.shape)

                    # print("global size",repA_public.size())
                    # print("local size", repA_avg_local.size())
                    loss_A = self.criterion_MSE(output_A, seq_A_public[:, inv_idx_public, :])
                    loss_B = self.criterion_MSE(output_B, seq_B_public[:, inv_idx_public, :])
                    lossKD = self.criterion_KL(embedding_knowledge_local,embedding_knowledge_global)
                    norm2loss = torch.dist(repA_public, repA_avg_local, p=2)
                    lossJSD = self.criterion_JSD(repA_public, repA_avg_local)
                    lossKD1 = self.criterion_KL(embedding_knowledge_local1,embedding_knowledge_global1)
                    # norm2loss1 = torch.dist(repA_public1, repA1_avg_local, p=2)
                    # lossJSD1 = self.criterion_JSD(repA_public1, repA1_avg_local)
                    lossSim = self.cos(repA_public, repA_avg_local)
                    # lossSim1 = self.cos(repA_public1, repA1_avg_local)
                    lossTrue_A = loss_A + loss_B


                        # loss = lossTrue  + eta * (lossKD + lossKD1)
                    if One_Layer:
                        lossA = lossTrue_A + eta*lossKD
                    else:
                        lossA = lossTrue_A + eta*lossKD + gamma*lossKD1
                    # loss = lossTrue + gamma * lossKD1
                    # Global_Knowledge_Transfer_Loss.append((lossKD1+lossKD).item())
                    Global_Knowledge_Transfer_Loss.append(lossKD.item())

                    # self.unfreeze(self.model.encoder_B)

                    # Train with input of modality B and output of modalities A&B
                    # self.freeze(self.model.encoder_A)
                    output_A, output_B = self.model(seq_B_public, "B")
                    loss_A = self.criterion_MSE(output_A, seq_A_public[:, inv_idx_public, :])
                    loss_B = self.criterion_MSE(output_B, seq_B_public[:, inv_idx_public, :])
                    lossKD = self.criterion_KL(embedding_knowledge_local,embedding_knowledge_global)
                    norm2loss = torch.dist(repB_public, repB_avg_local, p=2)
                    lossJSD = self.criterion_JSD(repB_public, repB_avg_local)
                    lossKD1 = self.criterion_KL(embedding_knowledge_local1,embedding_knowledge_global1)
                    # norm2loss1 = torch.dist(repB_public1, repB1_avg_local, p=2)
                    # lossJSD1 = self.criterion_JSD(repB_public1, repB1_avg_local)
                    lossSim = self.cos(repB_public, repB_avg_local)
                    # lossSim1 = self.cos(repB_public1, repB1_avg_local)
                    lossTrue_B = loss_A + loss_B
                    Global_ReconstructionLoss.append((lossTrue_A+lossTrue_B).item())


                        # loss = lossTrue  + eta * (lossKD + lossKD1)
                    if One_Layer:
                        lossB = lossTrue_B + eta*lossKD
                    else:
                        lossB = lossTrue_B + eta*lossKD + gamma*lossKD1
                    # loss = lossTrue + gamma * lossKD1
                    # Global_Knowledge_Transfer_Loss.append((lossKD1+lossKD).item())
                    Global_Knowledge_Transfer_Loss.append( lossKD.item())


                    loss = lossA+lossB
                    # loss = lossTrue_A + lossTrue_B + eta*lossKD + gamma*lossKD1
                    loss.backward()
                    self.optimizer_glob_ae.step()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


                    Global_Rec_loss.extend(Global_ReconstructionLoss)
                    Global_KT_Loss.extend(Global_Knowledge_Transfer_Loss)
            Global_Rec_round_loss.append(np.mean(Global_Rec_loss))
            Global_KT_round_loss.append(np.mean(Global_KT_Loss))
        # self.unfreeze(self.model)
        return np.mean(Global_Rec_round_loss), np.mean(Global_KT_round_loss)

    def global_update(self, epochs):
        # print("model param is",self.model.state_dict())
        # self.freeze(self.model)

        self.model.eval()
        self.model_server.train()
        # print("model server weight update is",self.model.state_dict())
        # print("server train A len is ", len(self.train_A["A"]))
        round_accuracy = []
        for epoch in range(1, epochs + 1):
            epoch_accuracy = []
            batch_size = np.random.randint(
                low=self.batch_min, high=self.batch_max)
            x_A_train, _, y_A_train = make_seq_batch(
                self.train_A, [0], len(self.train_A["A"]), batch_size)
            _, x_B_train, y_B_train = make_seq_batch(
                self.train_B, [0], len(self.train_B["B"]), batch_size)
            # x_A_train, _, y_A_train = make_seq_batch(
            #     self.public, [0], len(self.public["A"]), batch_size)
            # _, x_B_train, y_B_train = make_seq_batch(
            #     self.public, [0], len(self.public["B"]), batch_size)
            if "A" in label_modality:
                seq_len = x_A_train.shape[1]
                idx_start = 0
                idx_end = 0
                while idx_end < seq_len:
                    win_len = np.random.randint(low=16, high=32)
                    idx_start = idx_end
                    idx_end += win_len
                    idx_end = min(idx_end, seq_len)
                    x = x_A_train[:, idx_start:idx_end, :]
                    seq = torch.from_numpy(x).double().to(self.device)
                    y = y_A_train[:, idx_start:idx_end]

                    with torch.no_grad():
                        rpts,_ = self.model.encode(seq, "A")
                        # rpts = self.model.encode(seq, "A")
                    targets = torch.from_numpy(y.flatten()).to(self.device)
                    self.optimizer.zero_grad()
                    # print("representation size is",rpts.size())
                    output = self.model_server(rpts)
                    loss = self.criterion(output, targets.long())
                    top_p, top_class = output.topk(1, dim=1)
                    equals = top_class == targets.view(*top_class.shape).long()
                    accuracy = torch.mean(equals.type(torch.FloatTensor))

                    loss.backward()
                    self.optimizer.step()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    epoch_accuracy.append(accuracy)
            if "B" in label_modality:
                seq_len = x_B_train.shape[1]
                idx_start = 0
                idx_end = 0
                while idx_end < seq_len:
                    win_len = np.random.randint(low=16, high=32)
                    idx_start = idx_end
                    idx_end += win_len
                    idx_end = min(idx_end, seq_len)
                    x = x_B_train[:, idx_start:idx_end, :]
                    seq = torch.from_numpy(x).double().to(self.device)
                    y = y_B_train[:, idx_start:idx_end]

                    with torch.no_grad():
                        rpts,_ = self.model.encode(seq, "B")
                        # rpts= self.model.encode(seq, "B")
                    targets = torch.from_numpy(y.flatten()).to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model_server(rpts)
                    loss = self.criterion(output, targets.long())

                    top_p, top_class = output.topk(1, dim=1)
                    equals = top_class == targets.view(*top_class.shape).long()
                    accuracy = torch.mean(equals.type(torch.FloatTensor))
                    loss.backward()
                    self.optimizer.step()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    epoch_accuracy.append(accuracy)
            round_accuracy.append(np.mean(epoch_accuracy))

        print("Training accuracy is ", np.mean(round_accuracy))
        # self.unfreeze(self.model)
    def evaluate_local_encoder(self, user):

        user.model.eval()
        if Global_Classifier:
            self.model_server.eval()
        else:
            user.model_server.eval()
        if self.test_modality == "A":
            x_samples = np.expand_dims(self.data_test["A"], axis=0)
        elif self.test_modality == "B":
            x_samples = np.expand_dims(self.data_test["B"], axis=0)
        y_samples = np.expand_dims(self.data_test["y"], axis=0)
        win_loss = []
        win_accuracy = []
        win_f1 = []
        correct_pred=[]
        num_samples=[]
        n_samples = x_samples.shape[1]
        n_eval_process = n_samples // EVAL_WIN + 1

        for i in range(n_eval_process):
            idx_start = i * EVAL_WIN
            idx_end = np.min((n_samples, idx_start + EVAL_WIN))
            x = x_samples[:, idx_start:idx_end, :]
            y = y_samples[:, idx_start:idx_end]

            inputs = torch.from_numpy(x).double().to(self.device)
            targets = torch.from_numpy(y.flatten()).to(self.device)
            rpts, _ = user.model.encode(inputs, self.test_modality)
            # rpts= user.model.encode(inputs, self.test_modality)
            if Global_Classifier:
                output = self.model_server(rpts)
            else:
                output = user.model_server(rpts)

            loss = self.criterion(output, targets.long())
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            np_gt = y.flatten()
            np_pred = top_class.squeeze().cpu().detach().numpy()
            weighted_f1 = f1_score(np_gt, np_pred, average="weighted")

            win_loss.append(loss.item())
            win_accuracy.append(accuracy)
            win_f1.append(weighted_f1)
            correct_pred.append(np_pred)
            num_samples.append(np_gt)


            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.mean(win_f1)

    def client_encoder_test(self):
        local_f1_accuracy = []

        for user in self.selected_users:
            local_f1 = self.evaluate_local_encoder(user)
            local_f1_accuracy.append(local_f1)


        self.rs_local_f1_acc.append(local_f1_accuracy)
        return np.mean(local_f1_accuracy)

    def evaluating_encoder_clients(self):
        avg_test_accuracy= self.client_encoder_test()

        print("Avg clients F1 score is", avg_test_accuracy)
        return avg_test_accuracy

    def evaluating_classifier(self, epochs):
        # print("model param is", self.model_server.state_dict())
        self.model.eval()
        self.model_server.eval()
        # print(self.model_server.state_dict())
        if self.test_modality == "A":
            x_samples = np.expand_dims(self.data_test["A"], axis=0)
        elif self.test_modality == "B":
            x_samples = np.expand_dims(self.data_test["B"], axis=0)
        y_samples = np.expand_dims(self.data_test["y"], axis=0)

        win_loss = []
        win_accuracy = []
        win_f1 = []
        n_samples = x_samples.shape[1]
        n_eval_process = n_samples // EVAL_WIN + 1

        for i in range(n_eval_process):
            idx_start = i * EVAL_WIN
            idx_end = np.min((n_samples, idx_start+EVAL_WIN))
            x = x_samples[:, idx_start:idx_end, :]
            y = y_samples[:, idx_start:idx_end]

            inputs = torch.from_numpy(x).double().to(self.device)
            targets = torch.from_numpy(y.flatten()).to(self.device)
            rpts,_ = self.model.encode(inputs, self.test_modality)
            # rpts = self.model.encode(inputs, self.test_modality)
            output = self.model_server(rpts)

            loss = self.criterion(output, targets.long())
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            np_gt = y.flatten()
            np_pred = top_class.squeeze().cpu().detach().numpy()
            weighted_f1 = f1_score(np_gt, np_pred, average="weighted")

            win_loss.append(loss.item())
            win_accuracy.append(accuracy)
            win_f1.append(weighted_f1)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("loss is", np.mean(win_loss))
        print("F1 accuracy is",np.mean(win_f1))
        print("Test accuracy is",np.mean(win_accuracy))

        # print("representation", rpts)
        # print("Test accuracy is",np.mean(win_accuracy))
        self.rs_glob_acc.append(np.mean(win_f1))
        self.rs_test_loss.append(np.mean(win_loss))
        return np.mean(win_f1)
    def train(self):

        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            self.selected_users = self.select_users(self.num_glob_iters,self.num_users)
            # self.selected_users = self.users
            #
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)



            reconstruction_loss= []
            knowledge_transfer_loss = []
            local_f1_accuracy=[]
            #NOTE: this is required for the ``fork`` method to work
            #Update local autoencoder
            for user in self.selected_users:

                    # user.train(self.local_epochs)
                    pre_w = copy.deepcopy(user.model)
                    rec, kt=user.train_ae_distill(self.local_epochs,self.model,pre_w)
                    reconstruction_loss.append(rec)
                    knowledge_transfer_loss.append(kt)

            self.rs_rec_loss.append(reconstruction_loss)
            self.rs_kt_loss.append(knowledge_transfer_loss)

            #Update local classifier





            Global_rec,Global_kt=self.global_ae_distill(global_ae_distill_epoch,glob_iter)
            self.rs_global_rec_loss.append(Global_rec)
            self.rs_global_kt_loss.append(Global_kt)



            self.global_update(global_generalized_epochs)
            # Classifier evaluation
            if glob_iter % self.eval_interval != 0:
                continue
            else:
                with torch.no_grad():
                    self.evaluating_classifier(global_generalized_epochs)

            # if Global_Classifier==False:
            #     for user in self.selected_users:
            #         user.local_classifier_fine_tune(local_classifier_epochs,self.train_A,self.train_B)
            #
            # client_avg_acc=self.evaluating_encoder_clients()
            # self.c_avg_test.append(client_avg_acc)

            # for user in self.selected_users:
            #     local_f1 = self.evaluate_local_encoder(user)
            #     local_f1_accuracy.append(local_f1)
            # self.rs_local_f1_acc.append(local_f1_accuracy)
                # print("local f1 is", local_f1)
        # evaluate global classifier





        # self.save_results1()
        # self.save_model()
        self.save_results2()

    def save_results2(self):
        write_file(file_name=rs_file_path, root_test=self.rs_glob_acc, loss=self.rs_test_loss, rec_loss=self.rs_rec_loss, kt_loss=self.rs_kt_loss,
                   global_rec_loss=self.rs_global_rec_loss,global_kt_loss=self.rs_global_kt_loss,
                   N_clients=[N_clients])
        plot_from_file2()