import torch
import os
import torch.multiprocessing as mp
from tqdm import tqdm
import torch.nn as nn
from FLAlgorithms.users.userMultimodalRep import userMultimodalRep
from FLAlgorithms.users.userbase_dem import User
from FLAlgorithms.servers.serverbase_dem import Dem_Server
from Setting import rs_file_path, N_clients
from utils.data_utils import write_file
from utils.dem_plot import plot_from_file
from utils.model_utils import read_data, read_user_data, read_public_data,make_seq_batch,get_seg_len,load_data,client_idxs, split_public
from torch.utils.data import DataLoader
import numpy as np
from FLAlgorithms.optimizers.fedoptimizer import DemProx_SGD
from FLAlgorithms.trainmodel.models import *
# Implementation for Server
from utils.train_utils import KL_Loss, JSD, DCCLoss
from Setting import *
from torch import nn, optim
from sklearn.metrics import f1_score
class MultimodalRep(Dem_Server):
    def __init__(self, train_A, train_B,experiment, device, dataset, algorithm, model,  model_server, batch_size,
                 learning_rate, num_glob_iters, local_epochs, optimizer, num_users, times , cutoff,args):
        super().__init__(train_A, train_B,experiment, device, dataset, algorithm, model[0],  model_server[0],batch_size,
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
        self.eval_interval = 2
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
        # self.publicdatasetloader = DataLoader(read_public_data(dataset[0], dataset[1]), self.batch_size, shuffle=True)  # no shuffle
        # # self.publicloader= list(enumerate(self.publicdatasetloader))
        # self.enum_publicDS = enumerate(self.publicdatasetloader)
        # self.publicloader =[]
        # for b, (x, y) in self.enum_publicDS:
        #     self.publicloader.append((b,(x,y)))
        #     if(b<1): print(y)

        # self.publicdatasetlist= DataLoader(public_data, self.batch_size, shuffle=False)  # no shuffle
        sample=[]
        testing_sample=[]

        # while True:
        #     public_data = split_public(load_data(dataset)[2])
        #     if set(public_data["y"]) == set(server_test["y"]):
        #         break
        server_test = load_data(dataset)[1]
        public_data = load_data(dataset)[2]
        self.public = public_data
        self.data_test = server_test
        testing_sample = []
        modalities = ["A" for _ in range(self.num_clients_A)] + ["B" for _ in range(
            self.num_clients_B)] + ["AB" for _ in range(self.num_clients_AB)]
        for i in range(self.total_users):
            client_train = load_data(dataset)[0]
            # public_data = load_data(dataset)[2]
            id = client_idxs(client_train)

            # print("public len",len(public))
            # if(self.sub_data):
            #     if(i in randomList):
            #         train, test = self.get_data(train, test)
            user = userMultimodalRep(device, id[i], client_train, public_data,  model,  modalities[i], batch_size, learning_rate, beta,
                            local_epochs, optimizer)
            # self.publicloader = user.publicdatasetloader
            self.users.append(user)
            self.total_train_samples += user.train_samples

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


    def global_ae_distill(self, epochs):


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
            # if user.modality == "A " or user.modality == "AB":
            #     n_A += 1
            # if user.modality == "B" or user.modality == "AB":
            #     n_B += 1
            k+=1

        for user in self.selected_users:
            # batch_size = np.random.randint(
            #     low=self.batch_min, high=self.batch_max)
            a += 1
            b+=1
            repA_local.clear()
            repB_local.clear()
            repA1_local.clear()
            repB1_local.clear()
            if DATASET == "ur_fall":
                batch_size = 24
            else:
                batch_size = 200

            A_public, B_public, _ = make_seq_batch(
                self.public, [0], len(self.public["A"]), batch_size)
            seq_len_public = A_public.shape[1]
            idx_start_public = 0
            idx_end_public = 0

            while idx_end_public < seq_len_public:
                # win_len = np.random.randint(low=16, high=32)
                win_len = 24
                idx_start_public = idx_end_public
                idx_end_public += win_len
                idx_end_public = min(idx_end_public, seq_len_public)

                x_A_public = A_public[:, idx_start_public:idx_end_public, :]
                seq_A_public = torch.from_numpy(x_A_public).double().to(self.device)
                repA_public, repA_public1 = user.model.encode(seq_A_public, "A")
                repA_public = repA_public.cpu().detach().numpy()
                repA_public1 = repA_public1.cpu().detach().numpy()
                repA_local[idx_end_public]= repA_public
                repA1_local[idx_end_public] = repA_public1


                x_B_public = B_public[:, idx_start_public:idx_end_public, :]
                seq_B_public = torch.from_numpy(x_B_public).double().to(self.device)
                repB_public, repB_public1 = user.model.encode(seq_B_public, "B")
                repB_public = repB_public.cpu().detach().numpy()
                repB_public1 = repB_public1.cpu().detach().numpy()
                repB_local[idx_end_public] = repB_public
                repB1_local[idx_end_public] = repB_public1

            repA_local_full[a] = repA_local
            repB_local_full[b] = repB_local
            repA1_local_full[a] = repA1_local
            repB1_local_full[b] = repB1_local

        # print("rep B full", repB_local_full)
        n=0
        for client in repA_local_full.keys():
            repA = repA_local_full[client]
            for key in repA_local_full[client].keys():
                if (n==0):
                    repA_avg[key] = repA[key]/k
                else:
                    repA_avg[key] +=repA[key]/k
            n+=1

        m=0
        for client in repB_local_full.keys():
            repB = repB_local_full[client]
            for key in repB_local_full[client].keys():
                if (m == 0):
                    repB_avg[key] = repB[key] / k
                else:
                    repB_avg[key] += repB[key] / k
            m += 1
        g = 0
        for client in repA1_local_full.keys():
            repA1 = repA1_local_full[client]
            for key in repA1_local_full[client].keys():
                if (g == 0):
                    repA1_avg[key] = repA1[key] / k
                else:
                    repA1_avg[key] += repA1[key] / k
            g += 1

        j = 0
        for client in repB1_local_full.keys():
            repB1 = repB1_local_full[client]
            for key in repB1_local_full[client].keys():
                if (j == 0):
                    repB1_avg[key] = repB1[key] / k
                else:
                    repB1_avg[key] += repB1[key] / k
            j += 1
        # print("local rep B is", repB_avg)


        # TODO: Avg rep local output according num of samples of each users
        # TODO: Sum of DIR regularizer :done
        #
        # if self.gamma < 0.8 and glob_iter>=30:
        #     self.gamma += (0.8-0.5)/(NUM_GLOBAL_ITERS-30)
        #     # self.gamma += (0.5 - 0.05) / NUM_GLOBAL_ITERS



        #Global distillation
        # TODO: implement several global iterations to construct generalized knowledge :done
        for epoch in range(1, epochs+1):

            self.model.train()

            # for batch_idx, (X_public, y_public) in enumerate(self.publicloader):
            # batch_size = np.random.randint(
            #     low=self.batch_min, high=self.batch_max)
            if DATASET == "ur_fall":
                batch_size = 24
            else:
                batch_size = 200
            # print("batch size is", batch_size)
            A_public, B_public, _ = make_seq_batch(
                self.public, [0], len(self.public["A"]), batch_size)
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
                if self.model_ae == "split_LSTM":
                    self.freeze(self.model.encoder_B)
                    output_A, output_B = self.model(seq_A_public, "A")
                    repA_public, repA_public1 = self.model.encode(seq_A_public, "A")
                    repA_avg_local = torch.from_numpy(repA_avg[idx_end_public]).double().to(self.device)
                    repA1_avg_local = torch.from_numpy(repA1_avg[idx_end_public]).double().to(self.device)
                    # print("global size",repA_public.size())
                    # print("local size", repA_avg_local.size())
                    loss_A = self.criterion_MSE(output_A, seq_A_public[:, inv_idx_public, :])
                    loss_B = self.criterion_MSE(output_B, seq_B_public[:, inv_idx_public, :])
                    lossKD = self.criterion_KL(repA_public, repA_avg_local)
                    norm2loss = torch.dist(repA_public, repA_avg_local, p=2)
                    lossJSD = self.criterion_JSD(repA_public, repA_avg_local)
                    lossKD1 = self.criterion_KL(repA_public1, repA1_avg_local)
                    norm2loss1 = torch.dist(repA_public1, repA1_avg_local, p=2)
                    lossJSD1 = self.criterion_JSD(repA_public1, repA1_avg_local)

                    lossTrue = loss_A + loss_B
                    if Global_CDKT_metric == "KL":
                        loss = lossTrue + eta * (lossKD + lossKD1)
                    elif Global_CDKT_metric == "Norm2":
                        loss = lossTrue + eta *( norm2loss+norm2loss1)
                    elif Global_CDKT_metric == "JSD":
                        # print("doing here")
                        loss = lossTrue + eta * (lossJSD + lossJSD1)
                    loss.backward()
                    self.optimizer_glob_ae.step()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.unfreeze(self.model.encoder_B)

                    # Train with input of modality B and output of modalities A&B
                    self.freeze(self.model.encoder_A)
                    output_A, output_B = self.model(seq_B_public, "B")
                    repB_public, repB_public1 = self.model.encode(seq_B_public, "B")
                    repB_avg_local = torch.from_numpy(repB_avg[idx_end_public]).double().to(self.device)
                    repB1_avg_local = torch.from_numpy(repB1_avg[idx_end_public]).double().to(self.device)
                    loss_A = self.criterion_MSE(output_A, seq_A_public[:, inv_idx_public, :])
                    loss_B = self.criterion_MSE(output_B, seq_B_public[:, inv_idx_public, :])
                    lossKD = self.criterion_KL(repB_public, repB_avg_local)
                    norm2loss = torch.dist(repB_public, repB_avg_local, p=2)
                    lossJSD = self.criterion_JSD(repB_public, repB_avg_local)
                    lossKD1 = self.criterion_KL(repB_public1, repB1_avg_local)
                    norm2loss1 = torch.dist(repB_public1, repB1_avg_local, p=2)
                    lossJSD1 = self.criterion_JSD(repB_public1, repB1_avg_local)
                    lossTrue = loss_A + loss_B
                    if Global_CDKT_metric == "KL":
                        loss = lossTrue + gamma * (lossKD+lossKD1)
                    elif Global_CDKT_metric == "Norm2":
                        loss = lossTrue + gamma * (norm2loss + norm2loss1)
                    elif Global_CDKT_metric == "JSD":
                        loss = lossTrue + gamma * (lossJSD + lossJSD1)
                    loss.backward()
                    self.optimizer_glob_ae.step()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.unfreeze(self.model.encoder_B)
                elif self.model_ae == "DCCAE_LSTM":
                    rep_A, rep_B, output_A, output_B = self.model(x_A=seq_A_public, x_B=seq_B_public)
                    repA_public, repA_public1 = self.model.encode(seq_A_public, "A")
                    repA_avg_local = torch.from_numpy(repA_avg[idx_end_public]).double().to(self.device)
                    repA1_avg_local = torch.from_numpy(repA1_avg[idx_end_public]).double().to(self.device)
                    repB_public, repB_public1 = self.model.encode(seq_B_public, "B")
                    repB_avg_local = torch.from_numpy(repB_avg[idx_end_public]).double().to(self.device)
                    repB1_avg_local = torch.from_numpy(repB1_avg[idx_end_public]).double().to(self.device)
                    lossKD_A = self.criterion_KL(repA_public, repA_avg_local)
                    norm2loss_A = torch.dist(repA_public, repA_avg_local, p=2)
                    lossJSD_A = self.criterion_JSD(repA_public, repA_avg_local)
                    lossKD_B = self.criterion_KL(repB_public, repB_avg_local)
                    norm2loss_B = torch.dist(repB_public, repB_avg_local, p=2)
                    lossJSD_B = self.criterion_JSD(repB_public, repB_avg_local)
                    lossKD_A1 = self.criterion_KL(repA_public1, repA1_avg_local)
                    norm2loss_A1 = torch.dist(repA_public1, repA1_avg_local, p=2)
                    lossJSD_A1 = self.criterion_JSD(repA_public1, repA1_avg_local)
                    lossKD_B1 = self.criterion_KL(repB_public1, repB1_avg_local)
                    norm2loss_B1 = torch.dist(repB_public1, repB1_avg_local, p=2)
                    lossJSD_B1 = self.criterion_JSD(repB_public1, repB1_avg_local)
                    loss_A = self.criterion_MSE(output_A, seq_A_public[:, inv_idx_public, :])
                    loss_B = self.criterion_MSE(output_B, seq_B_public[:, inv_idx_public, :])
                    loss_dcc = self.criterion_DCC.loss(rep_A, rep_B)
                    if Global_CDKT_metric == "KL":
                        loss = loss_dcc + DCCAE_lamda * (loss_A + loss_B) + eta * (lossKD_A+lossKD_A1) + gamma * (lossKD_B+lossKD_B1)
                    elif Global_CDKT_metric == "Norm2":
                        loss = loss_dcc + DCCAE_lamda * (loss_A + loss_B) + eta * (norm2loss_A+ norm2loss_A1) + gamma * (norm2loss_B +norm2loss_B1)
                    elif Global_CDKT_metric == "JSD":
                        loss = loss_dcc + DCCAE_lamda * (loss_A + loss_B) + eta * (lossJSD_A +lossJSD_A1) + gamma * (lossJSD_B +   lossJSD_B1)

                    loss.backward()
                    self.optimizer_glob_ae.step()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()



    def global_update(self, epochs):
        # print("model param is",self.model.state_dict())
        self.freeze(self.model)
        self.model.eval()
        self.model_server.train()
        # print("model server weight update is",self.model.state_dict())

        round_accuracy = []
        for epoch in range(1, epochs + 1):
            epoch_accuracy = []
            batch_size = np.random.randint(
                low=self.batch_min, high=self.batch_max)
            x_A_train, _, y_A_train = make_seq_batch(
                self.train_A, [0], len(self.train_A["A"]), batch_size)
            _, x_B_train, y_B_train = make_seq_batch(
                self.train_B, [0], len(self.train_B["B"]), batch_size)
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
        self.unfreeze(self.model)
    def evaluating_classifier(self, glob_iter):
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

        print("F1 accuracy is",np.mean(win_f1))
        print("Test accuracy is",np.mean(win_accuracy))

    def train(self):
        for glob_iter in range(self.num_glob_iters):

            self.selected_users = self.select_users(glob_iter,self.num_users)
            # self.selected_users = self.users
            #
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")
            #
            # # ============= Test each client =============
            # tqdm.write('============= Test Client Models - Specialization ============= ')
            # stest_acu, strain_acc = self.evaluating_clients(glob_iter, mode="spe")
            # self.cs_avg_data_test.append(stest_acu)
            # self.cs_avg_data_train.append(strain_acc)
            # tqdm.write('============= Test Client Models - Generalization ============= ')
            # gtest_acu, gtrain_acc = self.evaluating_clients(glob_iter, mode="gen")
            # self.cg_avg_data_test.append(gtest_acu)
            # self.cg_avg_data_train.append(gtrain_acc)
            # tqdm.write('============= Test Global Models  ============= ')
            #loss_ = 0
            # self.send_parameters()   #Broadcast the global model to all clients


            #NOTE: this is required for the ``fork`` method to work
            for user in self.selected_users:

                    # user.train(self.local_epochs)

                    user.train_ae_distill(self.local_epochs,self.model)

                    # user.train_prox(self.local_epochs)

            #
            self.global_ae_distill(global_ae_distill_epoch)

            self.global_update(global_generalized_epochs)
            # Classifier evaluation
            if glob_iter % self.eval_interval != 0:
                continue
            else:
                with torch.no_grad():
                    self.evaluating_classifier(glob_iter)  # evaluate global classifier





        # self.save_results1()
        # self.save_model()

    def save_results1(self):
        write_file(file_name=rs_file_path, root_test=self.rs_glob_acc, root_train=self.rs_train_acc,
                   cs_avg_data_test=self.cs_avg_data_test, cs_avg_data_train=self.cs_avg_data_train,
                   cg_avg_data_test=self.cg_avg_data_test, cg_avg_data_train=self.cg_avg_data_train,
                   cs_data_test=self.cs_data_test, cs_data_train=self.cs_data_train, cg_data_test=self.cg_data_test,
                   cg_data_train=self.cg_data_train, N_clients=[N_clients])
        plot_from_file()
