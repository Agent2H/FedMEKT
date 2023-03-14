import json
import numpy as np
import os

import scipy.io
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import numpy as np
import random
from scipy.io import savemat, loadmat
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from Setting import *
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3
DATA_PATH =data_path
modality_a = MODALITY_A
modality_b = MODALITY_B
TRAIN_SUPERVISED_RATIO = train_supervised_ratio
TRAIN_RATIO = train_ratio
N_DIV_OPP = 100
N_DIV_MHEALTH = 100
N_DIV_URFALL = 10
N_LABEL_DIV_OPP = 15
N_LABEL_DIV_MHEALTH = 9
N_LABEL_DIV_URFALL = 9
NUM_LABELS = 0




def load_data(data):
    """Loads the dataset of the FL simulation.


    Args:
        config: a map of configurations of the simulation

    Returns:
        A dictionary containing training and testing data for modality A&B and labels.
    """
    save_dir = "./data/public_data"
    data = DATASET
    data_path = DATA_PATH
    modality_A = modality_a
    modality_B = modality_b

    if data == "opp":
        modalities = ["acce", "gyro"]
        assert (
            modality_A in modalities and modality_B in modalities), "Modality is neither acce nor gyro."
        mat_data = loadmat(os.path.join(data_path, "opp", "opp.mat"))

        data_train = {"A": zscore(mat_data[f"x_train_{modality_A}"]), "B": zscore(
            mat_data[f"x_train_{modality_B}"]), "y": np.squeeze(mat_data["y_train"])}
        data_test = {"A": zscore(mat_data[f"x_test_{modality_A}"]), "B": zscore(
            mat_data[f"x_test_{modality_B}"]), "y": np.squeeze(mat_data["y_test"])}
        data_public = {"A": zscore(mat_data[f"x_public_{modality_A}"]), "B": zscore(
            mat_data[f"x_public_{modality_B}"]), "y": np.squeeze(mat_data["y_public"])}

        # scipy.io.savemat('data_public_opp.mat', mdict=data_public)

        return (data_train, data_test, data_public)
    elif data == "mhealth":
        modalities = ["acce", "gyro", "mage"]
        assert (
            modality_A in modalities and modality_B in modalities), "Modality is not acce, gyro, or mage."
        mat_data = loadmat(os.path.join(data_path, "mhealth", "mhealth.mat"))
        # Randomely chooses 1 subject among all 10 subjects as testing data and the rest as training data
        s_test = np.random.randint(1, 11)
        # Create public data
        s_public = np.random.choice([i for i in range(1, 11) if i not in [s_test]])

        # print("s_test",s_test)
        # print("s_public",s_public)
        data_train = {"A": [], "B": [], "y": []}
        data_test = {}
        data_public = {}
        for i in range(1, 11):
            if i == s_test:
                data_test["A"] = zscore(mat_data[f"s{i}_{modality_A}"])
                data_test["B"] = zscore(mat_data[f"s{i}_{modality_B}"])
                data_test["y"] = np.squeeze(mat_data[f"s{i}_y"])
            elif i == s_public:
                data_public["A"] = zscore(mat_data[f"s{i}_{modality_A}"])
                data_public["B"] = zscore(mat_data[f"s{i}_{modality_B}"])
                data_public["y"] = np.squeeze(mat_data[f"s{i}_y"])
            else:
                data_train["A"].append(zscore(mat_data[f"s{i}_{modality_A}"]))
                data_train["B"].append(zscore(mat_data[f"s{i}_{modality_B}"]))
                data_train["y"].append(mat_data[f"s{i}_y"])
        data_train["A"] = np.concatenate(data_train["A"])
        data_train["B"] = np.concatenate(data_train["B"])
        data_train["y"] = np.squeeze(np.concatenate(data_train["y"], axis=1))
        # scipy.io.savemat('data_public_mhealth.mat', mdict=data_public)

        return (data_train, data_test, data_public)


    elif data == "ur_fall":
        modalities = ["acce", "rgb", "depth"]
        assert (
            modality_A in modalities and modality_B in modalities), "Modality is not acce, rgb, or depth."
        mat_data = loadmat(os.path.join(data_path, "ur_fall", "ur_fall.mat"))
        fall_test = np.random.choice(range(1, 31), 3, replace=False)
        adl_test = np.random.choice(range(1, 41), 4, replace=False)
        #Create public data
        fall_public = np.random.choice([i for i in range(1, 31) if i not in fall_test], 3, replace=False)
        adl_public = np.random.choice([i for i in range(1, 41) if i not in adl_test], 4, replace=False)
        data_train = {"A": [], "B": [], "y": []}
        data_test = {"A": [], "B": [], "y": []}
        data_public = {"A": [], "B": [], "y": []}
        a_A = mat_data[modality_A]
        a_B = mat_data[modality_B]
        a_y = mat_data["y"]

        for i in range(1, 31):
            sub_a_A = a_A[(a_A[:, 0] == 1) & (a_A[:, 1] == i), :]
            sub_a_B = a_B[(a_B[:, 0] == 1) & (a_B[:, 1] == i), :]
            sub_a_y = a_y[(a_y[:, 0] == 1) & (a_y[:, 1] == i), :]
            if modality_A == "acce" or modality_A == "depth":
                sub_a_A[:, 3:] = zscore(sub_a_A[:, 3:])
            if modality_B == "acce" or modality_B == "depth":
                sub_a_B[:, 3:] = zscore(sub_a_B[:, 3:])

            sub_a_A = sub_a_A[:, 3:]
            sub_a_B = sub_a_B[:, 3:]
            sub_a_y = sub_a_y[:, 3]

            if i in fall_test:
                data_test["A"].append(sub_a_A)
                data_test["B"].append(sub_a_B)
                data_test["y"].append(sub_a_y)
            elif i in fall_public:
                data_public["A"].append(sub_a_A)
                data_public["B"].append(sub_a_B)
                data_public["y"].append(sub_a_y)
            else:
                data_train["A"].append(sub_a_A)
                data_train["B"].append(sub_a_B)
                data_train["y"].append(sub_a_y)

        for i in range(1, 41):
            sub_a_A = a_A[(a_A[:, 0] == 0) & (a_A[:, 1] == i), :]
            sub_a_B = a_B[(a_B[:, 0] == 0) & (a_B[:, 1] == i), :]
            sub_a_y = a_y[(a_y[:, 0] == 0) & (a_y[:, 1] == i), :]
            if modality_A == "acce" or modality_A == "depth":
                sub_a_A[:, 3:] = zscore(sub_a_A[:, 3:])
            if modality_B == "acce" or modality_B == "depth":
                sub_a_B[:, 3:] = zscore(sub_a_B[:, 3:])

            sub_a_A = sub_a_A[:, 3:]
            sub_a_B = sub_a_B[:, 3:]
            sub_a_y = sub_a_y[:, 3]

            if i in adl_test:
                data_test["A"].append(sub_a_A)
                data_test["B"].append(sub_a_B)
                data_test["y"].append(sub_a_y)
            elif i in adl_public:
                data_public["A"].append(sub_a_A)
                data_public["B"].append(sub_a_B)
                data_public["y"].append(sub_a_y)
            else:
                data_train["A"].append(sub_a_A)
                data_train["B"].append(sub_a_B)
                data_train["y"].append(sub_a_y)

        data_train["A"] = np.concatenate(data_train["A"])
        data_train["B"] = np.concatenate(data_train["B"])
        data_train["y"] = np.squeeze(np.concatenate(data_train["y"]))
        data_test["A"] = np.concatenate(data_test["A"])
        data_test["B"] = np.concatenate(data_test["B"])
        data_test["y"] = np.squeeze(np.concatenate(data_test["y"]))
        data_public["A"] = np.concatenate(data_public["A"])
        data_public["B"] = np.concatenate(data_public["B"])
        data_public["y"] = np.squeeze(np.concatenate(data_public["y"]))

        # scipy.io.savemat('data_public_urfall.mat',mdict=data_public)
        return (data_train,  data_test, data_public)
        # return (data_train, data_test,data_public)

def split_server_train(data_train):
    """Extracts training data for the server.

    Args:
        data_train: a dictionary of training data of modalities A&B and labels y
        config: a map of configurations of the simulation

    Returns:
    A dictionary containing the server training data.
    """
    train_supervised_ratio = TRAIN_SUPERVISED_RATIO
    x_train_A = data_train["A"]
    x_train_B = data_train["B"]
    y_train = data_train["y"]
    server_train_A = np.empty((0, x_train_A.shape[1]))
    server_train_B = np.empty((0, x_train_B.shape[1]))
    server_train_y = np.empty((0))

    if DATASET == "opp":
        n_div = N_LABEL_DIV_OPP
    elif DATASET == "mhealth":
        n_div = N_LABEL_DIV_MHEALTH
    elif DATASET == "ur_fall":
        n_div = N_LABEL_DIV_URFALL
    n_server_train = round(n_div * train_supervised_ratio)
    n_row = len(x_train_A)
    n_sample_per_div = n_row // n_div
    idxs = np.arange(0, n_row, n_sample_per_div)
    slices_A = np.split(x_train_A, idxs)
    slices_B = np.split(x_train_B, idxs)
    slices_y = np.split(y_train, idxs)
    del slices_A[0]
    del slices_B[0]
    del slices_y[0]
    n_slices = len(slices_A)
    idxs_server_train = np.random.choice(
        np.arange(n_slices), n_server_train, replace=False)
    for i in range(n_slices):
        if i in idxs_server_train:
            server_train_A = np.concatenate((server_train_A, slices_A[i]))
            server_train_B = np.concatenate((server_train_B, slices_B[i]))
            server_train_y = np.concatenate((server_train_y, slices_y[i]))
    server_train = {"A": server_train_A,
                    "B": server_train_B, "y": server_train_y}
    return server_train

def split_public(data_public):
    """Extracts training data for the server.

    Args:
        data_train: a dictionary of training data of modalities A&B and labels y
        config: a map of configurations of the simulation

    Returns:
    A dictionary containing the server training data.
    """
    public_ratio = PUBLIC_RATIO
    x_public_A = data_public["A"]
    x_public_B = data_public["B"]
    y_public = data_public["y"]
    public_A = np.empty((0, x_public_A.shape[1]))
    public_B = np.empty((0, x_public_B.shape[1]))
    public_y = np.empty((0))

    if DATASET == "opp":
        n_div = N_LABEL_DIV_OPP
    elif DATASET == "mhealth":
        n_div = N_LABEL_DIV_MHEALTH
    elif DATASET == "ur_fall":
        n_div = N_LABEL_DIV_URFALL
    n_public = round(n_div * public_ratio)
    n_row = len(x_public_A)
    n_sample_per_div = n_row // n_div
    idxs = np.arange(0, n_row, n_sample_per_div)
    slices_A = np.split(x_public_A, idxs)
    slices_B = np.split(x_public_B, idxs)
    slices_y = np.split(y_public, idxs)
    del slices_A[0]
    del slices_B[0]
    del slices_y[0]
    n_slices = len(slices_A)
    idxs_public = np.random.choice(
        np.arange(n_slices), n_public, replace=False)
    for i in range(n_slices):
        if i in idxs_public:
            public_A = np.concatenate((public_A, slices_A[i]))
            public_B = np.concatenate((public_B, slices_B[i]))
            public_y = np.concatenate((public_y, slices_y[i]))
    public_data = {"A": public_A,
                    "B": public_B, "y": public_y}
    return public_data

def make_seq_batch(dataset, seg_idxs, seg_len, batch_size):
    """Makes batches of sequences from the dataset.

    Args:
        dataset: a dictionary containing data of modalities A&B and labels y
        seg_idxs: A list containing the starting indices of the segments in all samples for a client.
        seg_len: An integer indicating the length of a segment
        batch_size: An integer indicating the number of batches

    Returns:
        A tuple containing the batches of sequences of modalities A&B and labels y
    """
    samples_A = dataset["A"]
    samples_B = dataset["B"]
    samples_y = dataset["y"]

    input_size_A = len(samples_A[0])
    input_size_B = len(samples_B[0])
    # the length of each sequence
    seq_len = seg_len * len(seg_idxs) // batch_size
    if seq_len > seg_len:
        seq_len = seg_len - 1

    all_indices_start = []
    for idx in seg_idxs:
        indices_start_in_seg = list(range(idx, idx + seg_len - seq_len))
        all_indices_start.extend(indices_start_in_seg)
    indices_start = np.random.choice(
        all_indices_start, batch_size, replace=False)

    A_seq = np.zeros((batch_size, seq_len, input_size_A), dtype=np.float32)
    B_seq = np.zeros((batch_size, seq_len, input_size_B), dtype=np.float32)
    y_seq = np.zeros((batch_size, seq_len), dtype=np.uint8)

    for i in range(batch_size):
        idx_start = indices_start[i]
        idx_end = idx_start+seq_len
        A_seq[i, :, :] = samples_A[idx_start: idx_end, :]
        B_seq[i, :, :] = samples_B[idx_start: idx_end, :]
        y_seq[i, :] = samples_y[idx_start:idx_end]
    return (A_seq, B_seq, y_seq)

def get_seg_len(n_samples):
    if DATASET== "opp":
        n_div = N_DIV_OPP
    elif DATASET == "mhealth":
        n_div = N_DIV_MHEALTH
    elif DATASET == "ur_fall":
        n_div = N_DIV_URFALL
    return int(n_samples * float(train_ratio)//n_div)

def get_seg_len_public(n_samples):
    if DATASET== "opp":
        n_div = N_DIV_OPP
    elif DATASET == "mhealth":
        n_div = N_DIV_MHEALTH
    elif DATASET == "ur_fall":
        n_div = N_DIV_URFALL
    return int(n_samples * float(PUBLIC_RATIO))

def client_idxs(data_train):
    """Generates sample indices for each client.

    Args:
        data_train: a dictionary containing training data of modalities A&B and labels y
        config: a map of configurations of the simulation

    Returns:
    A list containing the sample indices for each client. Each item in the list is a list of numbers and each number representing the starting location of a segment in the training data.
    """
    num_clients_A = NUM_CLIENT_A
    num_clients_B = NUM_CLIENT_B
    num_clients_AB = NUM_CLIENT_AB
    num_clients = num_clients_A+num_clients_B+num_clients_AB

    n_samples = len(data_train["A"])  # number of rows in training data
    # divide the training data into divisions
    if DATASET == "opp":
        n_div = N_DIV_OPP
    elif DATASET == "mhealth":
        n_div = N_DIV_MHEALTH
    elif DATASET == "ur_fall":
        n_div = N_DIV_URFALL
    # each client has (n_samples * train_ratio) data
    train_ratio = TRAIN_RATIO

    len_div = int(n_samples // n_div)  # the length of each division
    # Within each division, we randomly pick 1 segment. So the length of each segment is
    len_seg = get_seg_len(n_samples)
    starts_div = np.arange(0, n_samples-len_div, len_div)
    idxs_clients = []
    for i in range(num_clients):
        idxs_clients.append(np.array([]).astype(np.int64))
        for start in starts_div:
            idxs_in_div = np.arange(start, start + len_div - len_seg)
            idxs_clients[i] = np.append(
                idxs_clients[i], np.random.choice(idxs_in_div))
    return idxs_clients
class Metrics(object):
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics_dir = os.path.join('out', self.params['dataset'], 'metrics_{}_{}_{}_{}_{}.json'.format(
            self.params['seed'], self.params['optimizer'], self.params['learning_rate'], self.params['num_epochs'], self.params['mu']))
        #os.mkdir(os.path.join('out', self.params['dataset']))
        if not os.path.exists('out'):
            os.mkdir('out')
        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)
