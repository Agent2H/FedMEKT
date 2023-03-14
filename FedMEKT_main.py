#!/usr/bin/env python
from comet_ml import Experiment
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serverMultimodalMoon import MultimodalMoon
from FLAlgorithms.servers.serverMultimodalFedAvg import MultimodalFedAvg
from FLAlgorithms.servers.serverMultimodalFedProx import MultimodalFedProx
from FLAlgorithms.servers.serverFedMEKTS import MultimodalRep
from FLAlgorithms.servers.serverFedMEKTC import MultimodalRepFusion
from utils.model_utils import  load_data, split_server_train
from FLAlgorithms.trainmodel.ae_model import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)
from utils.options import args_parser
import warnings
warnings.filterwarnings('ignore')
# import comet_ml at the top of your file
#python CDKT_main.py --dataset Mnist --model cnn --learning_rate 0.03 --num_global_iters 200  --algorithm FedAvg --times 1 --subusers 0.1
#python CDKT_main.py --dataset Mnist --model cnn --learning_rate 0.03 --num_global_iters 200  --algorithm --times 1 --subusers 0.1

#python CDKT_main.py --dataset Mnist --model cnn --learning_rate 0.03 --num_global_iters 200  --algorithm --times 1 --subusers 1

# Create an experiment with your api key:
def main(experiment, dataset, algorithm, model, model_server, embedding_layer, embedding_layer1, batch_size, learning_rate, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, commet, gpu, cutoff, args):

    # print torch.device()
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    # data = read_data(dataset) , dataset
    data = load_data(dataset)

    server_test = data[1]
    while True:
            train_A = split_server_train(data[0])
            if set(train_A["y"]) == set(server_test["y"]):
                break
    while True:
            train_B = split_server_train(data[0])
            if set(train_B["y"]) == set(server_test["y"]):
                break

    input_size_A = train_A["A"].shape[1]
    input_size_B = train_B["B"].shape[1]
    n_classes = len(set(train_A["y"]))
    # print("rep size is", rep_size)
    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model


                    # model = SplitLSTMAutoEncoder(input_size_A, input_size_B, rep_size).double().to(device), model
                    # model = SplitLSTMAutoEncoder_Embedding(input_size_A, input_size_B, rep_size).double().to(device), model
        model = SplitLSTMAutoEncoder2(input_size_A, input_size_B, rep_size).double().to(device), model
        # print("model is",model)

        if (model_server == "MLP"):
                    if (dataset == "mhealth"):
                        model_server = MLP(rep_size,n_classes).double().to(device), model_server
                    else:
                        model_server = MLP(rep_size, n_classes).double().to(device), model_server
                        # model_server = MLP(Num_neurons, n_classes).double().to(device), model_server
        if embedding_layer=="Dense":
            embedding_layer = Dense(rep_size).double().to(device), embedding_layer
        if embedding_layer1 == "Dense1":
            embedding_layer1 = Dense1(Num_neurons).double().to(device), embedding_layer1

        # select algorithm

        if (algorithm == "mmFedAvg"):
            if (commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(
                    learning_rate) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers))
            server = MultimodalFedAvg(train_A,train_B, experiment, device, data, algorithm, model,  model_server, embedding_layer,embedding_layer1, batch_size, learning_rate,
                          num_glob_iters, local_epochs, optimizer, numusers, i, cutoff, args)

        elif (algorithm == "mmFedProx"):
            if (commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(
                    learning_rate) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers))
            server = MultimodalFedProx(train_A, train_B, experiment, device, data, algorithm, model, model_server,
                                      embedding_layer, embedding_layer1, batch_size, learning_rate,
                                      num_glob_iters, local_epochs, optimizer, numusers, i, cutoff, args)
        elif (algorithm == "mmMOON"):
            if (commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(
                    learning_rate) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers))
            server = MultimodalMoon(train_A, train_B, experiment, device, data, algorithm, model, model_server,
                                       embedding_layer, embedding_layer1, batch_size, learning_rate,
                                       num_glob_iters, local_epochs, optimizer, numusers, i, cutoff, args)

        elif (algorithm == "FedMEKT-S"):
            if (commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(
                    learning_rate) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers))
            server = MultimodalRep(train_A, train_B, experiment, device, data, algorithm, model, model_server, embedding_layer,embedding_layer1,
                                      batch_size, learning_rate, num_glob_iters, local_epochs, optimizer, numusers, i, cutoff, args)
        elif (algorithm == "FedMEKT-C"):
            if (commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(
                    learning_rate) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers))
            server = MultimodalRepFusion(train_A, train_B, experiment, device, data, algorithm, model, model_server,
                                   embedding_layer, embedding_layer1,
                                   batch_size, learning_rate, num_glob_iters, local_epochs, optimizer, numusers, i,
                                   cutoff, args)




        else:
            print("Algorithm is invalid")
            return

        server.train()
        # server.test()

    # average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L_k,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times, cutoff = cutoff)

if __name__ == "__main__":
    args = args_parser()
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))

    print("Learing rate       : {}".format(args.learning_rate))
    print("Subset of users      : {}".format(args.subusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    # print("Server Model       : {}".format(args.server_model))

    print("=" * 80)

    if(args.commet):
        # Create an experiment with your api key:
        experiment = Experiment(
            api_key="VtHmmkcG2ngy1isOwjkm5sHhP",
            project_name="multitask-for-test",
            workspace="federated-learning-exp",
        )

        hyper_params = {
            "dataset":args.dataset,
            "algorithm" : args.algorithm,
            "model":args.model,
            # "server_model":args.server_model,
            "client_model": args.client_model,
            "batch_size":args.batch_size,
            "learning_rate":args.learning_rate,
            "beta" : args.beta,
            "L_k" : args.L_k,
            "num_glob_iters":args.num_global_iters,
            "local_epochs":args.local_epochs,
            "optimizer": args.optimizer,
            "numusers": args.subusers,
            "K" : args.K,
            "personal_learning_rate" : args.personal_learning_rate,
            "times" : args.times,
            "gpu": args.gpu,
            "cut-off": args.cutoff
        }

        experiment.log_parameters(hyper_params)
    else:
        experiment = 0

    main(
        experiment= experiment,
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        model_server=args.model_server,
        embedding_layer=args.embedding_layer,
        embedding_layer1=args.embedding_layer1,
        # server_model=args.server_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,

        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.subusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times = args.times,
        commet = args.commet,
        gpu=args.gpu,
        cutoff = args.cutoff,
        args=args
        )


