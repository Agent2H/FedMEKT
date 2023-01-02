#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
from Setting import *

def args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser = argparse.ArgumentParser()
    if(DATASET == "mnist"):
        Input_DS = "Mnist"
    else:
        Input_DS = DATASET
    if(RUNNING_ALG == "fedavg"):
        Input_Alg = "FedAvg"
    else:
        Input_Alg = RUNNING_ALG

    parser.add_argument("--dataset", type=str, default=Input_DS, choices=["mhealth","opp","ur_fall"])
    # parser.add_argument("--server_model", type=str, default="cnn", choices=["cnn","resnet"])
    parser.add_argument("--algorithm", type=str, default=Input_Alg, choices=["mmFedAvg","mmFedProx","FedMEKT","FedEKD","FedMEFKT"])
    parser.add_argument("--model", type=str, default=MODEL_AE, choices=["split_LSTM", "DCCAE_LSTM"])
    parser.add_argument("--model_server", type=str, default="MLP", choices=[ "MLP"])
    parser.add_argument("--embedding_layer", type=str, default="Dense", choices=["Dense"])
    parser.add_argument("--embedding_layer1", type=str, default="Dense1", choices=["Dense1"])
    parser.add_argument("--label_modality", type=str, default=label_modality, choices=["A","B"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=local_learning_rate, help="Local learning rate")
    parser.add_argument("--classifier_learning_rate", type=float, default=0.01, help="classifier learning rate")
    parser.add_argument("--DCCAE_lambda", type=float, default=0.01, help="lambda factor for DCCAE")
    parser.add_argument("--num_global_iters", type=int, default=NUM_GLOBAL_ITERS)
    parser.add_argument("--local_epochs", type=int, default = LOCAL_EPOCH)
    parser.add_argument("--classifier_epochs", type=int, default = 5)
    parser.add_argument("--optimizer", type=str, default="SGD")

    parser.add_argument("--subusers", type = float, default = Frac_users, help="Fraction of Num Users per round")  #Fraction number of users
    parser.add_argument("--K", type=int, default=0, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.02, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--commet", type=int, default=0, help="log data to commet")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments") #GPU dev_id, -1 is CPU
    parser.add_argument("--cutoff", type=int, default=0, help="Cutoff data sample")
    parser.add_argument("--DECAY", type=bool, default=0, help="DECAY or CONSTANT")
    parser.add_argument("--mu", type=int, default=0, help="mu parameter")
    parser.add_argument("--gamma", type=int, default=0, help="gama parameter")
    parser.add_argument("--total_users", type=int, default=N_clients, help="total users")
    args = parser.parse_args()

    return args
