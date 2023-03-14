import torch
import os
import torch.multiprocessing as mp
from tqdm import tqdm
import torch.nn as nn
from FLAlgorithms.users.userMultimodalFedProx import UserMultimodalFedProx
from FLAlgorithms.users.userbase import User
# from FLAlgorithms.servers.serverbase_dem import Dem_Server
from FLAlgorithms.servers.serverbase import Server
from Setting import rs_file_path, N_clients
from utils.data_utils import write_file
from utils.dem_plot import plot_from_file, plot_from_file2
from utils.model_utils import make_seq_batch,get_seg_len,load_data,client_idxs
from torch.utils.data import DataLoader
import numpy as np


# Implementation for Server
from utils.train_utils import KL_Loss, JSD
from Setting import *
from torch import nn, optim
from sklearn.metrics import f1_score
class MultimodalFedProx(Server):
    def __init__(self, train_A, train_B, experiment, device, dataset, algorithm, model, model_server,embedding_layer,embedding_layer1, batch_size, learning_rate,
                  num_glob_iters, local_epochs, optimizer, num_users, times, cutoff, args):
        super().__init__(train_A, train_B, experiment, device, dataset, algorithm, model[0],model_server[0],embedding_layer[0],embedding_layer1[0],  batch_size, learning_rate,
                          num_glob_iters, local_epochs, optimizer, num_users, times, args)

        # Initialize data for all  users
        if DATASET == "ur_fall":
            self.batch_min = 16
            self.batch_max = 32
        else:
            self.batch_min = 128
            self.batch_max = 256
        self.train_A = train_A
        self.train_B = train_B
        self.eval_interval =1
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model_server.parameters(), lr=global_learning_rate)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion_KL = KL_Loss(temperature=3.0)
        self.criterion_JSD = JSD()
        self.test_modality = test_modality
        self.gamma = gamma
        self.num_clients_A = NUM_CLIENT_A
        self.num_clients_B = NUM_CLIENT_B
        self.num_clients_AB = NUM_CLIENT_AB
        self.save_interval=2
        # total_users = len(dataset[0][0])
        self.sub_data = cutoff
        if (self.sub_data):
            randomList = self.get_partion(self.total_users)



        # self.publicdatasetlist= DataLoader(public_data, self.batch_size, shuffle=False)  # no shuffle
        sample = []
        server_test = load_data(dataset)[1]
        public = load_data(dataset)[2]
        # print('public len', len(public["y"]))
        self.public = public
        self.data_test = server_test
        testing_sample = []
        modalities = ["A" for _ in range(self.num_clients_A)] + ["B" for _ in range(
            self.num_clients_B)] + ["AB" for _ in range(self.num_clients_AB)]
        for i in range(self.total_users):
            client_train = load_data(dataset)[0]
            id = client_idxs(client_train)
            public_data = load_data(dataset)[2]
            # print("User ", id, ": Numb of Training data", len(train))
            # sample.append(len(train) + len(test))
            # print("public len",len(public))
            # if (self.sub_data):
            #     if (i in randomList):
            #         train, test = self.get_data(train, test)
            user = UserMultimodalFedProx(device, id[i], client_train, public_data, model,  modalities[i], batch_size, learning_rate, beta,
                            local_epochs, optimizer)

            self.users.append(user)
            self.total_train_samples += user.train_samples

            # print(user.train_samples)
        # print("train sample median :", np.median(training_sample))
        # print("test sample median :", np.median(testing_sample))
        print("sample is", np.median(sample))

        # self.local_model = user.local_model
        # self.train_samples = len(client_train)

        print("Fraction number of users / total users:", num_users, " / ", self.total_users)

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


    def global_update(self, epochs):

        self.model.eval()
        self.model_server.train()
        # print("model server update 1 is", self.model.state_dict())

        round_accuracy = []
        for epoch in range(epochs):
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
                # print("output len is", seq_len)
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
                        # rpts= self.model.encode(seq, "A")
                    targets = torch.from_numpy(y.flatten()).to(self.device)
                    self.optimizer.zero_grad()
                    # print("representation size is",rpts.size())
                    output= self.model_server(rpts)
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
                        # rpts = self.model.encode(seq, "B")
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
        print("Training accuracy is ",np.mean(round_accuracy))

    # def evaluate_local_encoder(self, user):
    #     user.model.eval()
    #     self.model_server.eval()
    #     if self.test_modality == "A":
    #         x_samples = np.expand_dims(self.data_test["A"], axis=0)
    #     elif self.test_modality == "B":
    #         x_samples = np.expand_dims(self.data_test["B"], axis=0)
    #     y_samples = np.expand_dims(self.data_test["y"], axis=0)
    #     win_loss = []
    #     win_accuracy = []
    #     win_f1 = []
    #     n_samples = x_samples.shape[1]
    #     n_eval_process = n_samples // EVAL_WIN + 1
    #
    #     for i in range(n_eval_process):
    #         idx_start = i * EVAL_WIN
    #         idx_end = np.min((n_samples, idx_start + EVAL_WIN))
    #         x = x_samples[:, idx_start:idx_end, :]
    #         y = y_samples[:, idx_start:idx_end]
    #
    #         inputs = torch.from_numpy(x).double().to(self.device)
    #         targets = torch.from_numpy(y.flatten()).to(self.device)
    #         rpts, _ = user.model.encode(inputs, self.test_modality)
    #         # rpts= user.model.encode(inputs, self.test_modality)
    #         output = self.model_server(rpts)
    #
    #         loss = self.criterion(output, targets.long())
    #         top_p, top_class = output.topk(1, dim=1)
    #         equals = top_class == targets.view(*top_class.shape).long()
    #         accuracy = torch.mean(equals.type(torch.FloatTensor))
    #         np_gt = y.flatten()
    #         np_pred = top_class.squeeze().cpu().detach().numpy()
    #         weighted_f1 = f1_score(np_gt, np_pred, average="weighted")
    #
    #         win_loss.append(loss.item())
    #         win_accuracy.append(accuracy)
    #         win_f1.append(weighted_f1)
    #
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
    #     return np.mean(win_f1)
    #
    # def client_encoder_test(self):
    #     local_f1_accuracy = []
    #
    #     for user in self.selected_users:
    #         local_f1 = self.evaluate_local_encoder(user)
    #         local_f1_accuracy.append(local_f1)
    #
    #
    #     self.rs_local_f1_acc.append(local_f1_accuracy)
    #     return np.mean(local_f1_accuracy)
    #
    # def evaluating_encoder_clients(self):
    #     avg_test_accuracy = self.client_encoder_test()
    #
    #     print("Avg clients F1 score is", avg_test_accuracy)
    #     return avg_test_accuracy

    def evaluating_classifier(self, epochs):
        self.model.eval()
        self.model_server.eval()
        # print("model server update 2 is",self.model.state_dict())

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
            # rpts= self.model.encode(inputs, self.test_modality)
            output = self.model_server(rpts)

            loss = self.criterion(output, targets.long())
            # print("evaluation loss is",loss)
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            np_gt = y.flatten()
            np_pred = top_class.squeeze().cpu().detach().numpy() #PRINT OUT
            weighted_f1 = f1_score(np_gt, np_pred, average="weighted")

            win_loss.append(loss.item())
            win_accuracy.append(accuracy)
            win_f1.append(weighted_f1)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("loss is" , np.mean(win_loss))
        # print("representation", rpts)
        print("F1 accuracy is",np.mean(win_f1))
        # print("Test accuracy is",np.mean(win_accuracy))
        self.rs_glob_acc.append(np.mean(win_f1))
        self.rs_test_loss.append(np.mean(win_loss))
        return np.mean(win_f1)

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            self.selected_users = self.select_users(self.num_glob_iters, self.num_users)

            if (self.experiment):
                self.experiment.set_epoch(glob_iter + 1)


            # loss_ = 0
            self.send_parameters()   #Broadcast the global model to all clients

            #
            # # NOTE: this is required for the ``fork`` method to work
            reconstruction_loss = []
            local_f1_accuracy = []
            for user in self.selected_users:

                    rec=user.train_ae(self.local_epochs,self.model)
                    reconstruction_loss.append(rec)
            # print("model server",self.model.state_dict())

            # print("model server", self.model.state_dict())
            #
            self.rs_rec_loss.append(reconstruction_loss)

            self.aggregate_parameters_multimodal()  # Aggregate parameters from local autoencoder



            self.global_update(global_generalized_epochs) #update global classifier with representation from global ae
            # train_acc = self.global_update(global_generalized_epochs)
            # tqdm.write('At round {} AvgC. training accuracy: {}'.format(glob_iter, train_acc))
            # print("model server after update", self.model.state_dict())

            #Classifier evaluation
            if glob_iter % self.eval_interval !=0:
                continue
            else:
                with torch.no_grad():
                    self.evaluating_classifier(global_generalized_epochs)  # evaluate global classifier
            #
            # client_avg_acc = self.evaluating_encoder_clients()
            # self.c_avg_test.append(client_avg_acc)


        # self.save_results1()
        # self.save_model()
        self.save_results2()

    def save_results2(self):
        write_file(file_name=rs_file_path, root_test=self.rs_glob_acc, loss=self.rs_test_loss, rec_loss=self.rs_rec_loss,
                   local_f1_acc=self.rs_local_f1_acc,avg_local_f1_acc=self.c_avg_test,N_clients=[N_clients])
        plot_from_file2()