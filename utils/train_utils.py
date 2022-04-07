from torchvision import datasets, transforms
from FLAlgorithms.trainmodel.models import *
import torch.nn.functional as F
import numpy as np
from Setting import *



class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss


class JSD(nn.Module):

    def __init__(self):
        super(JSD, self).__init__()

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)

        m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0

        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), m, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), m, reduction="batchmean")

        return (0.5 * loss)






class DCCLoss():
    def __init__(self, outdim_size, device, use_all_singular_values=False):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        # print(device)

    def loss(self, H1, H2):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)
#         print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
#         print(Tval.size())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            # regularization for more stability
            trace_TT = torch.add(trace_TT, (torch.eye(
                trace_TT.shape[0])*r1).to(self.device))
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U > eps, U, (torch.ones(
                U.shape).double()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr

def get_model(args):
    if(args.model == "mclr"):
        if(args.dataset == "human_activity"):
            model = Mclr_Logistic(561,6).to(args.device)
        elif(args.dataset == "gleam"):
            model = Mclr_Logistic(561,6).to(args.device)
        elif(args.dataset == "vehicle_sensor"):
            model = Mclr_Logistic(100,2).to(args.device)
        elif(args.dataset == "Synthetic"):
            model = Mclr_Logistic(60,10).to(args.device)
        else:#(dataset == "Mnist"):
            model = Mclr_Logistic().to(args.device)

    elif(args.model == "dnn"):
        if(args.dataset == "human_activity"):
            model = DNN(561,100,12).to(args.device)
        elif(args.dataset == "gleam"):
            model = DNN(561,20,6).to(args.device)
        elif(args.dataset == "vehicle_sensor"):
            model = DNN(100,20,2).to(args.device)
        elif(args.dataset == "Synthetic"):
            model = DNN(60,20,10).to(args.device)
        else:#(dataset == "Mnist"):
            model = DNN2().to(args.device)
        
    elif(args.model == "cnn"):
        if(args.dataset == "Cifar10"):
            model = CNNCifar().to(args.device)
    else:
        pasexit('Error: unrecognized model')
    return model