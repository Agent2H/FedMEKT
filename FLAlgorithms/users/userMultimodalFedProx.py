import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
# from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import DemProx_SGD, FedProxOptimizer
from FLAlgorithms.users.userbase_dem import User
from FLAlgorithms.trainmodel.models import *
from torch import Tensor
from collections import OrderedDict
# Implementation for clients
from utils.train_utils import KL_Loss, JSD, DCCLoss
from utils.model_utils import read_data, read_user_data, read_public_data, make_seq_batch, get_seg_len
from Setting import *


class UserMultimodalFedProx(User):
    def __init__(self, device, client_train_idx, client_train,public_data, model,  modality, batch_size, learning_rate,
                 beta, local_epochs, optimizer):
        super().__init__(device, client_train_idx, client_train,public_data, model[0],  modality, batch_size,
                         learning_rate, beta, local_epochs)


        self.model_ae = MODEL_AE
        self.modality = modality
        self.train = client_train
        self.train_idx = client_train_idx
        self.seg_len = get_seg_len(len(client_train["A"]))
        self.loss = nn.CrossEntropyLoss()
        self.criterion_KL = KL_Loss(temperature=3.0)
        self.criterion_JSD = JSD()
        self.criterion_MSE = nn.MSELoss().to(self.device)
        self.rep_size = rep_size
        self.criterion_DCC = DCCLoss(self.rep_size, self.device)
        self.alpha = ALPHA

        if DATASET == "ur_fall":
            self.batch_min = 16
            self.batch_max = 32
        else:
            self.batch_min = 128
            self.batch_max = 256

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=local_learning_rate)
        # self.optimizer = FedProxOptimizer(self.model.parameters(), lr=local_learning_rate)

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

    def get_weight(self):
        """Gets the training weight of the client"""
        # Since all clients have same amount of local data, it's same as local
        # data size=1 for all. So the weight for multimodal clients is ALPHA and
        # the weight for unimodal clients is 1.
        if self.modality == "AB":
            return ALPHA
        else:
            return 1


    def train_ae(self, epochs,global_model):

        self.model.train()
        gen_model = copy.deepcopy(global_model)
        gen_model.eval()
        for param in gen_model.parameters():
            param.requires_grad = False
        round_loss = []
        Rec_round_loss = []
        KT_round_loss=[]
        for epoch in range(epochs):
            Rec_loss = []
            KT_Loss = []
            epoch_losses = []

            batch_size = np.random.randint(
                low=self.batch_min, high=self.batch_max)
            A_train, B_train, _ = make_seq_batch(
                self.train, self.train_idx, self.seg_len, batch_size)
            # A_train and B_train both are in the shape of (batch_size, seq_len, input_size), i.e., batch first
            seq_len = A_train.shape[1]


            idx_start = 0
            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(low=16, high=32)
                idx_start = idx_end
                idx_end += win_len
                idx_end = min(idx_end, seq_len)

                if self.model_ae== "split_LSTM":
                    ReconstructionLoss = []
                    Knowledge_Transfer_Loss = []
                    if self.modality == "A" or self.modality == "AB":
                        x_A = A_train[:, idx_start:idx_end, :]
                        seq_A = torch.from_numpy(x_A).double().to(self.device)
                        inv_idx = torch.arange(seq_A.shape[1] - 1, -1, -1).long()

                    if self.modality == "B" or self.modality == "AB":
                        x_B = B_train[:, idx_start:idx_end, :]
                        seq_B = torch.from_numpy(x_B).double().to(self.device)
                        inv_idx = torch.arange(seq_B.shape[1] - 1, -1, -1).long()
                    self.optimizer.zero_grad()
                    if self.modality == "A":
                        # print("doing here")
                        self.freeze(self.model.encoder_B)
                        self.freeze(self.model.decoder_B)

                        output, _ = self.model(seq_A, "A")
                        loss = self.criterion_MSE(output, seq_A[:, inv_idx, :])

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
                        loss = self.criterion_MSE(output, seq_B[:, inv_idx, :])

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
                        proximal_termA = 0.0
                        for w, w_t in zip(self.model.encoder_A.parameters(), gen_model.encoder_A.parameters()):
                            proximal_termA += (w - w_t).norm(2)


                        loss_A = self.criterion_MSE(output_A, seq_A[:, inv_idx, :])
                        loss_B = self.criterion_MSE(output_B, seq_B[:, inv_idx, :])
                        loss = loss_A + loss_B+ 0.01*proximal_termA
                        ReconstructionLoss.append(loss.item())

                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.unfreeze(self.model.encoder_B)

                        # Train with input of modality B and output of modalities A&B
                        self.freeze(self.model.encoder_A)
                        output_A, output_B = self.model(seq_B, "B")
                        proximal_termB = 0.0
                        for w, w_t in zip(self.model.encoder_B.parameters(), gen_model.encoder_B.parameters()):
                            proximal_termB += (w - w_t).norm(2)
                        loss_A = self.criterion_MSE(output_A, seq_A[:, inv_idx, :])
                        loss_B = self.criterion_MSE(output_B, seq_B[:, inv_idx, :])

                        loss = loss_A + loss_B+0.01*proximal_termB
                        ReconstructionLoss.append(loss.item())
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

                    if self.modality == "B" or self.modality == "AB":
                        x_B = B_train[:, idx_start:idx_end, :]
                        seq_B = torch.from_numpy(x_B).double().to(self.device)
                        inv_idx = torch.arange(seq_B.shape[1] - 1, -1, -1).long()
                    # Train with input of modalities A&B and output of modalities A&B
                    self.optimizer.zero_grad()
                    if self.modality == "A":
                        self.freeze(self.model.encoder_B)
                        self.freeze(self.model.decoder_B)

                        _, _, output, _ = self.model(x_A=seq_A)
                        loss = self.criterion_MSE(output, seq_A[:, inv_idx, :])
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
                        loss = self.criterion_MSE(output, seq_B[:, inv_idx, :])
                        # sub_epoch_losses.append(loss.item())
                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        self.unfreeze(self.model.encoder_A)
                        self.unfreeze(self.model.decoder_A)
                    elif self.modality == "AB":
                        rep_A, rep_B, output_A, output_B = self.model(x_A=seq_A, x_B=seq_B)
                        loss_A = self.criterion_MSE(output_A, seq_A[:, inv_idx, :])
                        loss_B = self.criterion_MSE(output_B, seq_B[:, inv_idx, :])
                        loss_dcc = self.criterion_DCC.loss(rep_A, rep_B)
                        loss = loss_dcc + DCCAE_lamda * (loss_A + loss_B)
                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                Rec_loss.extend(ReconstructionLoss)

            Rec_round_loss.append(np.mean(Rec_loss))

        return np.mean(Rec_round_loss)

        # self.clone_model_paramenter(self.model.parameters(), self.local_model)  # update the local model with new local weight
