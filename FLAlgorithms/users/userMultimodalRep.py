import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
# from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import DemProx_SGD
from FLAlgorithms.users.userbase_dem import User
from FLAlgorithms.trainmodel.models import *
from torch import Tensor
from collections import OrderedDict
# Implementation for clients
from utils.train_utils import KL_Loss, JSD, DCCLoss
from utils.model_utils import read_data, read_user_data, read_public_data, make_seq_batch, get_seg_len
from Setting import *

class userMultimodalRep(User):
    def __init__(self, device, client_train_idx, client_train, public_data, model,  modality, batch_size, learning_rate,
                 beta, local_epochs, optimizer):
        super().__init__(device, client_train_idx, client_train,public_data, model[0],  modality, batch_size,
                         learning_rate, beta, local_epochs)

        # if(model[1] == "Mclr_CrossEntropy"):
        #     self.loss = nn.CrossEntropyLoss()
        # else:
        #     self.loss = nn.NLLLoss()
        self.model_ae = MODEL_AE
        self.modality = modality
        self.train = client_train
        self.public = public_data
        self.train_idx = client_train_idx
        self.seg_len = get_seg_len(len(client_train["A"]))
        self.seg_len_public = get_seg_len(len(public_data["A"]))
        self.loss = nn.CrossEntropyLoss()
        self.criterion_KL = KL_Loss(temperature=3.0)
        self.criterion_JSD = JSD()
        self.criterion_MSE = nn.MSELoss().to(self.device)
        self.rep_size = rep_size
        self.criterion_DCC = DCCLoss(self.rep_size, self.device)
        if DATASET == "ur_fall":
            self.batch_min = 16
            self.batch_max = 32
        else:
            self.batch_min = 128
            self.batch_max = 256
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=local_learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # if Same_model:
        #     if Accelerated:
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=local_learning_rate)
        #     else:
        #         self.optimizer = DemProx_SGD(self.model.parameters(), lr=local_learning_rate, mu=0)
        #
        # else:
        #     self.optimizer = DemProx_SGD(self.client_model.parameters(), lr=local_learning_rate, mu=0)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def freeze(self, sub_model):
        """Freeze the parameters of a model"""
        for param in sub_model.parameters():
            param.requires_grad = False

    def unfreeze(self, sub_model):
        """Unfreeze the parameters of a model"""
        for param in sub_model.parameters():
            param.requires_grad = True

    def train_ae_distill(self, epochs, global_model):

        gen_model = copy.deepcopy(global_model)
        gen_model.train()
        self.model.train()
        round_loss = []
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            self.model.train()
            batch_size = np.random.randint(
                low=self.batch_min, high=self.batch_max)

            A_train, B_train, _ = make_seq_batch(
                self.train, self.train_idx, self.seg_len, batch_size)
            # A_train and B_train both are in the shape of (batch_size, seq_len, input_size), i.e., batch first
            seq_len = A_train.shape[1]
            # print("public len is ",len(self.public["A"]))
            A_public, B_public, _ = make_seq_batch(
                self.public, [0], len(self.public["A"]), batch_size)
            seq_len_public = A_public.shape[1]
            idx_start = 0
            idx_end = 0
            idx_start_public = 0
            idx_end_public = 0
            while idx_end < seq_len and idx_end_public < seq_len_public:
                win_len = np.random.randint(low=16, high=32)
                idx_start = idx_end
                idx_end += win_len
                idx_end = min(idx_end, seq_len)

                idx_start_public = idx_end_public
                idx_end_public += win_len
                idx_end_public = min(idx_end_public, seq_len_public)


                if self.model_ae== "split_LSTM":
                    if self.modality == "A" or self.modality == "AB":
                        x_A = A_train[:, idx_start:idx_end, :]
                        seq_A = torch.from_numpy(x_A).double().to(self.device)
                        inv_idx = torch.arange(seq_A.shape[1] - 1, -1, -1).long()

                        x_A_public = A_public[:, idx_start_public:idx_end_public, :]
                        seq_A_public = torch.from_numpy(x_A_public).double().to(self.device)
                        inv_idx_public = torch.arange(seq_A_public.shape[1] - 1, -1, -1).long()

                    if self.modality == "B" or self.modality == "AB":
                        x_B = B_train[:, idx_start:idx_end, :]
                        seq_B = torch.from_numpy(x_B).double().to(self.device)
                        inv_idx = torch.arange(seq_B.shape[1] - 1, -1, -1).long()

                        x_B_public = B_public[:, idx_start_public:idx_end_public, :]
                        seq_B_public = torch.from_numpy(x_B_public).double().to(self.device)
                        inv_idx_public = torch.arange(seq_B_public.shape[1] - 1, -1, -1).long()
                    self.optimizer.zero_grad()
                    if self.modality == "A":
                        # print("doing here")
                        self.freeze(self.model.encoder_B)
                        self.freeze(self.model.decoder_B)

                        output, _ = self.model(seq_A, "A")
                        repA_public, repA_public1 = self.model.encode(seq_A_public,"A")
                        gen_repA_public, gen_repA_public1 = gen_model.encode(seq_A_public,"A")
                        lossKD= self.criterion_KL(repA_public, gen_repA_public)
                        lossKD1 = self.criterion_KL(repA_public1, gen_repA_public1)
                        norm2loss = torch.dist(repA_public, gen_repA_public, p=2)
                        norm2loss1 = torch.dist(repA_public1, gen_repA_public1, p=2)
                        lossJSD = self.criterion_JSD(repA_public,gen_repA_public)
                        lossJSD1 = self.criterion_JSD(repA_public1, gen_repA_public1)
                        lossTrue = self.criterion_MSE(output, seq_A[:, inv_idx, :])
                        if Local_CDKT_metric == "KL":
                            loss = lossTrue + alpha * lossKD + alpha*lossKD1
                        elif Local_CDKT_metric == "Norm2":
                            loss = lossTrue + alpha * norm2loss + alpha*norm2loss1
                        elif Local_CDKT_metric == "JSD":
                            loss = lossTrue + alpha * lossJSD + alpha*lossJSD1

                        # sub_epoch_losses.append(loss.item())
                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        self.unfreeze(self.model.encoder_B)
                        self.unfreeze(self.model.decoder_B)
                    elif self.modality == "B":
                        # print("doing here")
                        self.freeze(self.model.encoder_A)
                        self.freeze(self.model.decoder_A)

                        _, output = self.model(seq_B, "B")
                        repB_public, repB_public1 = self.model.encode(seq_B_public, "B")
                        gen_repB_public, gen_repB_public1 = gen_model.encode(seq_B_public, "B")
                        lossKD = self.criterion_KL(repB_public, gen_repB_public)
                        lossKD1 = self.criterion_KL(repB_public1, gen_repB_public1)
                        norm2loss = torch.dist(repB_public, gen_repB_public, p=2)
                        norm2loss1 = torch.dist(repB_public1, gen_repB_public1, p=2)
                        lossJSD = self.criterion_JSD(repB_public, gen_repB_public)
                        lossJSD1 = self.criterion_JSD(repB_public1, gen_repB_public1)
                        lossTrue = self.criterion_MSE(output, seq_B[:, inv_idx, :])
                        if Local_CDKT_metric == "KL":
                            loss = lossTrue + beta * lossKD + beta * lossKD1
                        elif Local_CDKT_metric == "Norm2":
                            loss = lossTrue + beta * norm2loss + beta * norm2loss1
                        elif Local_CDKT_metric == "JSD":
                            loss = lossTrue + beta * lossJSD + beta * lossJSD1

                        # sub_epoch_losses.append(loss.item())
                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        self.unfreeze(self.model.encoder_A)
                        self.unfreeze(self.model.decoder_A)
                    elif self.modality == "AB":
                        # Train with input of modality A and output of modalities A&B
                        # print("doing here")
                        self.freeze(self.model.encoder_B)
                        output_A, output_B = self.model(seq_A, "A")
                        repA_public, repA_public1 = self.model.encode(seq_A_public,"A")
                        gen_repA_public, gen_repA_public1 = gen_model.encode(seq_A_public,"A")
                        loss_A = self.criterion_MSE(output_A, seq_A[:, inv_idx, :])
                        loss_B = self.criterion_MSE(output_B, seq_B[:, inv_idx, :])
                        lossKD = self.criterion_KL(repA_public, gen_repA_public)
                        lossKD1 = self.criterion_KL(repA_public1, gen_repA_public1)
                        norm2loss = torch.dist(repA_public, gen_repA_public, p=2)
                        norm2loss1 = torch.dist(repA_public1, gen_repA_public1, p=2)
                        lossJSD = self.criterion_JSD(repA_public, gen_repA_public)
                        lossJSD1 = self.criterion_JSD(repA_public1, gen_repA_public1)

                        lossTrue = loss_A + loss_B
                        if Local_CDKT_metric == "KL":
                            loss = lossTrue + alpha * lossKD + alpha * lossKD1
                        elif Local_CDKT_metric == "Norm2":
                            loss = lossTrue + alpha * norm2loss + alpha * norm2loss1
                        elif Local_CDKT_metric == "JSD":
                            loss = lossTrue + alpha * lossJSD + alpha * lossJSD1

                        # print("loss is",loss)
                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.unfreeze(self.model.encoder_B)

                        # Train with input of modality B and output of modalities A&B
                        self.freeze(self.model.encoder_A)
                        output_A, output_B = self.model(seq_B, "B")
                        repB_public, repB_public1 = self.model.encode(seq_B_public, "B")
                        gen_repB_public, gen_repB_public1 = gen_model.encode(seq_B_public, "B")
                        loss_A = self.criterion_MSE(output_A, seq_A[:, inv_idx, :])
                        loss_B = self.criterion_MSE(output_B, seq_B[:, inv_idx, :])
                        lossKD = self.criterion_KL(repB_public, gen_repB_public)
                        lossKD1 = self.criterion_KL(repB_public1, gen_repB_public1)
                        norm2loss = torch.dist(repB_public, gen_repB_public, p=2)
                        norm2loss1 = torch.dist(repB_public1, gen_repB_public1, p=2)
                        lossJSD = self.criterion_JSD(repB_public, gen_repB_public)
                        lossJSD1 = self.criterion_JSD(repB_public1, gen_repB_public1)
                        lossTrue = loss_A + loss_B
                        if Local_CDKT_metric == "KL":
                            loss = lossTrue + beta * lossKD + beta * lossKD1
                        elif Local_CDKT_metric == "Norm2":
                            loss = lossTrue + beta * norm2loss + beta * norm2loss1
                        elif Local_CDKT_metric == "JSD":
                            loss = lossTrue + beta * lossJSD + beta * lossJSD1


                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.unfreeze(self.model.encoder_A)
                elif self.model_ae == "DCCAE_LSTM":
                    if self.modality == "A" or self.modality == "AB":
                        x_A = A_train[:, idx_start:idx_end, :]
                        seq_A = torch.from_numpy(x_A).double().to(self.device)
                        inv_idx = torch.arange(seq_A.shape[1] - 1, -1, -1).long()

                        x_A_public = A_public[:, idx_start_public:idx_end_public, :]
                        seq_A_public = torch.from_numpy(x_A_public).double().to(self.device)
                        inv_idx_public = torch.arange(seq_A_public.shape[1] - 1, -1, -1).long()

                    if self.modality == "B" or self.modality == "AB":
                        x_B = B_train[:, idx_start:idx_end, :]
                        seq_B = torch.from_numpy(x_B).double().to(self.device)
                        inv_idx = torch.arange(seq_B.shape[1] - 1, -1, -1).long()

                        x_B_public = B_public[:, idx_start_public:idx_end_public, :]
                        seq_B_public = torch.from_numpy(x_B_public).double().to(self.device)
                        inv_idx_public = torch.arange(seq_B_public.shape[1] - 1, -1, -1).long()
                    # Train with input of modalities A&B and output of modalities A&B
                    self.optimizer.zero_grad()
                    if self.modality == "A":
                        self.freeze(self.model.encoder_B)
                        self.freeze(self.model.decoder_B)

                        _, _, output, _ = self.model(x_A=seq_A)
                        repA_public, repA_public1 = self.model.encode(seq_A_public, "A")
                        gen_repA_public, gen_repA_public1 = gen_model.encode(seq_A_public, "A")
                        lossKD = self.criterion_KL(repA_public, gen_repA_public)
                        lossKD1 = self.criterion_KL(repA_public1, gen_repA_public1)
                        norm2loss = torch.dist(repA_public, gen_repA_public, p=2)
                        norm2loss1 = torch.dist(repA_public1, gen_repA_public1, p=2)
                        lossJSD = self.criterion_JSD(repA_public, gen_repA_public)
                        lossJSD1 = self.criterion_JSD(repA_public1, gen_repA_public1)
                        lossTrue = self.criterion_MSE(output, seq_A[:, inv_idx, :])
                        if Local_CDKT_metric == "KL":
                            loss = lossTrue + alpha * lossKD + alpha * lossKD1
                        elif Local_CDKT_metric == "Norm2":
                            loss = lossTrue + alpha * norm2loss + alpha * norm2loss1
                        elif Local_CDKT_metric == "JSD":
                            loss = lossTrue + alpha * lossJSD + alpha * lossJSD1
                        # sub_epoch_losses.append(loss.item())
                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        self.unfreeze(self.model.encoder_B)
                        self.unfreeze(self.model.decoder_B)
                    elif self.modality == "B":
                        self.freeze(self.model.encoder_A)
                        self.freeze(self.model.decoder_A)

                        _, _, _, output = self.model(x_B=seq_B)
                        repB_public, repB_public1 = self.model.encode(seq_B_public, "B")
                        gen_repB_public, gen_repB_public1 = gen_model.encode(seq_B_public, "B")
                        lossKD = self.criterion_KL(repB_public, gen_repB_public)
                        lossKD1 = self.criterion_KL(repB_public1, gen_repB_public1)
                        norm2loss = torch.dist(repB_public, gen_repB_public, p=2)
                        norm2loss1 = torch.dist(repB_public1, gen_repB_public1, p=2)
                        lossJSD = self.criterion_JSD(repB_public, gen_repB_public)
                        lossJSD1 = self.criterion_JSD(repB_public1, gen_repB_public1)
                        lossTrue = self.criterion_MSE(output, seq_B[:, inv_idx, :])
                        if Local_CDKT_metric == "KL":
                            loss = lossTrue + beta * lossKD + beta * lossKD1
                        elif Local_CDKT_metric == "Norm2":
                            loss = lossTrue + beta * norm2loss + beta * norm2loss1
                        elif Local_CDKT_metric == "JSD":
                            loss = lossTrue + beta * lossJSD + beta * lossJSD1
                        # sub_epoch_losses.append(loss.item())
                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        self.unfreeze(self.model.encoder_A)
                        self.unfreeze(self.model.decoder_A)
                    elif self.modality == "AB":
                        rep_A, rep_B, output_A, output_B = self.model(x_A=seq_A, x_B=seq_B)
                        repA_public, repA_public1 = self.model.encode(seq_A_public, "A")
                        gen_repA_public, gen_repA_public1 = gen_model.encode(seq_A_public, "A")
                        repB_public, repB_public1 = self.model.encode(seq_B_public, "B")
                        gen_repB_public, gen_repB_public1 = gen_model.encode(seq_B_public, "B")
                        lossKD_A = self.criterion_KL(repA_public, gen_repA_public)
                        norm2loss_A = torch.dist(repA_public, gen_repA_public, p=2)
                        lossJSD_A = self.criterion_JSD(repA_public, gen_repA_public)
                        lossKD_B = self.criterion_KL(repB_public, gen_repB_public)
                        norm2loss_B = torch.dist(repB_public, gen_repB_public, p=2)
                        lossJSD_B = self.criterion_JSD(repB_public, gen_repB_public)
                        lossKD_A1 = self.criterion_KL(repA_public1, gen_repA_public1)
                        norm2loss_A1 = torch.dist(repA_public1, gen_repA_public1, p=2)
                        lossJSD_A1 = self.criterion_JSD(repA_public1, gen_repA_public1)
                        lossKD_B1 = self.criterion_KL(repB_public1, gen_repB_public1)
                        norm2loss_B1 = torch.dist(repB_public1, gen_repB_public1, p=2)
                        lossJSD_B1 = self.criterion_JSD(repB_public1, gen_repB_public1)
                        loss_A = self.criterion_MSE(output_A, seq_A[:, inv_idx, :])
                        loss_B = self.criterion_MSE(output_B, seq_B[:, inv_idx, :])
                        loss_dcc = self.criterion_DCC.loss(rep_A, rep_B)
                        if Local_CDKT_metric == "KL":
                            loss = loss_dcc + DCCAE_lamda * (loss_A + loss_B) + alpha * (lossKD_A+lossKD_A1) +beta*(lossKD_B+lossKD_B1)
                        elif Local_CDKT_metric == "Norm2":
                            loss = loss_dcc + DCCAE_lamda * (loss_A + loss_B) + alpha * (norm2loss_A+norm2loss_A1) + beta*(norm2loss_B+norm2loss_B1)
                        elif Local_CDKT_metric == "JSD":
                            loss = loss_dcc + DCCAE_lamda * (loss_A + loss_B) + alpha * lossJSD_A + beta*lossJSD_B

                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()




