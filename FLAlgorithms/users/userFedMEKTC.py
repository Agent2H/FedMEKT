import copy

import numpy as np
import torch
import torch.nn as nn


from FLAlgorithms.users.userbase import User

# Implementation for clients
from utils.train_utils import KL_Loss, JSD, DCCLoss
from utils.model_utils import make_seq_batch, get_seg_len,split_public,get_seg_len_public
from Setting import *

class userMultimodalRepFusion(User):
    def __init__(self, device, client_train_idx, client_train, public_data, model, model_server, embedding_layer,embedding_layer1,  modality, batch_size, learning_rate,
                 beta, local_epochs, optimizer):
        super().__init__(device, client_train_idx, client_train,public_data, model[0], model_server[0], embedding_layer[0],embedding_layer1[0],  modality, batch_size,
                         learning_rate, beta, local_epochs)


        self.model_ae = MODEL_AE
        self.modality = modality
        self.train = client_train
        self.public = public_data
        self.train_idx = client_train_idx
        self.seg_len = get_seg_len(len(client_train["A"]))
        self.seg_len_public = get_seg_len(len(public_data["A"]))
        self.seg_len_public_test = get_seg_len_public(len(public_data["A"]))
        self.loss = nn.CrossEntropyLoss()
        self.criterion_KL = KL_Loss(temperature=3.0)
        self.criterion_JSD = JSD()
        self.criterion_MSE = nn.MSELoss().to(self.device)
        self.cos = nn.CosineSimilarity(dim=1)
        self.rep_size = rep_size
        self.criterion_DCC = DCCLoss(self.rep_size, self.device)
        self.temperature =0.5
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # self.public_test=split_public(self.public)



        if DATASET == "ur_fall":
            self.batch_min = 16
            self.batch_max = 32
        else:
            self.batch_min = 128
            self.batch_max = 256
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=local_learning_rate)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

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

    def train_ae_distill(self, epochs, global_model,pre_w):

        gen_model = copy.deepcopy(global_model)
        gen_model.eval()
        for param in gen_model.parameters():
            param.requires_grad = False

        prev_model = copy.deepcopy(pre_w)
        prev_model.eval()
        for param in prev_model.parameters():
            param.requires_grad = False

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        Rec_round_loss = []
        KT_round_loss=[]
        for epoch in range(1, epochs + 1):
            Rec_loss = []
            KT_Loss =[]
            # self.model.train()
            batch_size = np.random.randint(
                low=self.batch_min, high=self.batch_max)

            A_train, B_train, _ = make_seq_batch(
                self.train, self.train_idx, self.seg_len, batch_size)
            # A_train and B_train both are in the shape of (batch_size, seq_len, input_size), i.e., batch first
            seq_len = A_train.shape[1]

            #
            # A_public, B_public, _ = make_seq_batch(
            #     self.public, [0], len(self.public["A"]), batch_size)
            # seq_len_public = A_public.shape[1]


            A_public, B_public, _ = make_seq_batch(
                self.public, [0], self.seg_len_public_test, batch_size)
            seq_len_public = A_public.shape[1]

            # print("private seg len", self.seg_len)
            # print("private seq ",seq_len)
            # print("public seg len", len(self.public["A"]))
            # print("public seq ", seq_len_public)
            # print("public len client is ",len(self.public["A"]))
            # print("private len is ", len(self.train["A"]))

            # print("public len is ",len(self.public_test["A"]))
            idx_start = 0
            idx_end = 0
            idx_start_public = 0


            idx_end_public = 0
            while idx_end < seq_len and idx_end_public<seq_len_public :
                    win_len = np.random.randint(low=16, high=32)
                    idx_start = idx_end
                    idx_end += win_len
                    idx_end = min(idx_end, seq_len)

                    idx_start_public = idx_end_public
                    idx_end_public += win_len
                    idx_end_public = min(idx_end_public, seq_len_public)
                    # if idx_end>seq_len and idx_end_public<seq_len_public:
                    #     idx_end=0
                    #     idx_end += win_len


                    ReconstructionLoss=[]
                    Knowledge_Transfer_Loss=[]

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
                        self.freeze(self.model.encoder_B)
                        self.freeze(self.model.decoder_B)

                        output, _ = self.model(seq_A, "A")
                        repA_public, repA_public1 = self.model.encode(seq_A_public, "A")
                        gen_repA_public, gen_repA_public1 = gen_model.encode(seq_A_public, "A")
                        lossKD = self.criterion_KL(repA_public, gen_repA_public)
                        lossKD1 = self.criterion_KL(repA_public1, gen_repA_public1)
                        lossTrue = self.criterion_MSE(output, seq_A[:, inv_idx, :])

                        loss = lossTrue + alpha * lossKD + beta*lossKD1

                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        self.unfreeze(self.model.encoder_B)
                        self.unfreeze(self.model.decoder_B)
                    elif self.modality == "B":
                        self.freeze(self.model.encoder_A)
                        self.freeze(self.model.decoder_A)
                        _, output = self.model(seq_B, "B")
                        repB_public, repB_public1 = self.model.encode(seq_B_public, "B")
                        gen_repB_public, gen_repB_public1 = gen_model.encode(seq_B_public, "B")
                        lossKD = self.criterion_KL(repB_public, gen_repB_public)
                        lossKD1 = self.criterion_KL(repB_public1, gen_repB_public1)
                        lossTrue = self.criterion_MSE(output, seq_B[:, inv_idx, :])

                        loss = lossTrue + alpha * lossKD + beta * lossKD1

                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        self.unfreeze(self.model.encoder_A)
                        self.unfreeze(self.model.decoder_A)
                    elif self.modality == "AB":
                    # Train with input of modality A and output of modalities A&B

                        repA, repA1 = self.model.encode(seq_A_public, "A")
                        repB, repB1 = self.model.encode(seq_B_public, "B")

                        gen_repA, gen_repA1 = gen_model.encode(seq_A_public, "A")
                        gen_repB, gen_repB1 = gen_model.encode(seq_B_public, "B")

                        embedding_knowledge_local = torch.cat([repA, repB], dim=0)
                        embedding_knowledge_local = self.embedding_layer(embedding_knowledge_local)
                        embedding_knowledge_global = torch.cat([gen_repA, gen_repB], dim=0)
                        embedding_knowledge_global = self.embedding_layer(embedding_knowledge_global)
                        embedding_knowledge_local1 = torch.cat([repA1, repB1], dim=0)
                        embedding_knowledge_local1 = self.embedding_layer1(embedding_knowledge_local1)
                        embedding_knowledge_global1 = torch.cat([gen_repA1, gen_repB1], dim=0)
                        embedding_knowledge_global1 = self.embedding_layer1(embedding_knowledge_global1)

                        output_A, output_B = self.model(seq_A, "A")
                        loss_A = self.criterion_MSE(output_A, seq_A[:, inv_idx, :])
                        loss_B = self.criterion_MSE(output_B, seq_B[:, inv_idx, :])
                        # loss_A_proxy = self.criterion_MSE(output_A_public, seq_A_public[:, inv_idx_public, :])
                        # loss_B_proxy = self.criterion_MSE(output_B_public, seq_B_public[:, inv_idx_public, :])
                        lossKD = self.criterion_KL(embedding_knowledge_local, embedding_knowledge_global)
                        lossKD1 = self.criterion_KL(embedding_knowledge_local1, embedding_knowledge_global1)
                        # print("loss kd",lossKD)

                        lossTrue_A = loss_A + loss_B
                        # ReconstructionLoss.append(lossTrue.item())


                        # loss = lossTrue + alpha * (lossKD +  lossKD1)
                        if One_Layer:
                            lossA = lossTrue_A + alpha * lossKD
                        else:
                            lossA = lossTrue_A + alpha * lossKD +  beta*lossKD1
                            # loss = lossTrue + alpha * lossKD + beta * lossKD1+ lamda*lossTrueproxy

                            # Knowledge_Transfer_Loss.append((lossKD1+lossKD).item())
                            Knowledge_Transfer_Loss.append( lossKD.item())
                        output_A, output_B = self.model(seq_B, "B")


                        # Loss function
                        loss_A = self.criterion_MSE(output_A, seq_A[:, inv_idx, :])
                        loss_B = self.criterion_MSE(output_B, seq_B[:, inv_idx, :])

                        lossKD = self.criterion_KL(embedding_knowledge_local, embedding_knowledge_global)
                        lossKD1 = self.criterion_KL(embedding_knowledge_local1,embedding_knowledge_global1)



                        lossTrue_B = loss_A + loss_B
                        ReconstructionLoss.append((lossTrue_A+lossTrue_B).item())

                            # loss = lossTrue + alpha * (lossKD + lossKD1)
                        if One_Layer:
                            lossB = lossTrue_B + alpha * lossKD
                        else:
                            lossB = lossTrue_B + alpha * lossKD + beta * lossKD1

                            # Knowledge_Transfer_Loss.append((lossKD1+lossKD).item())
                        Knowledge_Transfer_Loss.append( lossKD.item())

                        loss= lossA  + lossB
                        # loss = lossTrue_A +  lossTrue_B+ alpha*lossKD+ beta*lossKD1
                        loss.backward()
                        self.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()


                        Rec_loss.extend(ReconstructionLoss)
                        KT_Loss.extend(Knowledge_Transfer_Loss)

            Rec_round_loss.append(np.mean(Rec_loss))
            KT_round_loss.append(np.mean(KT_Loss))
        # print ("rec loss is", np.mean(Rec_round_loss))
        # print ("kt loss is", np.mean(KT_round_loss))
        return np.mean(Rec_round_loss), np.mean(KT_round_loss)









