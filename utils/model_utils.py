import json
import numpy as np
import os
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

def suffer_data(data):
    data_x = data['x']
    data_y = data['y']
        # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)
    
def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts +1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x,data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)

def read_cifa_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, train_data in enumerate(testloader,0):
        testset.data, testset.targets = train_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 20 # should be muitiple of 10
    NUM_LABELS = 3
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_image.extend(testset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_label.extend(testset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)

    cifa_data = []
    for i in trange(10):
        idx = cifa_data_label==i
        cifa_data.append(cifa_data_image[idx])


    print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []

    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            print("L:", l)
            X[user] += cifa_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()
            idx[l] += 10

    print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in cifa_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    #idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples)  + numran1 #+ 200
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(cifa_data[l]):
                X[user] += cifa_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                # print("check len os user:", user, j,
                #     "len data", len(X[user]), num_samples)

    print("IDX2:", idx) # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        
        
    # random.seed(1)
    # np.random.seed(1)
    # NUM_USERS = 1 # should be muitiple of 10
    # NUM_LABELS = 10
    # # Setup directory for train/test data
    # cifa_data_image = []
    # cifa_data_label = []

    # cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    # cifa_data_label.extend(trainset.targets.cpu().detach().numpy())

    # cifa_data_image = np.array(cifa_data_image)
    # cifa_data_label = np.array(cifa_data_label)

    # # Create data structure
    # train_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # # Setup 5 users
    # # for i in trange(5, ncols=120):
    # for i in range(NUM_USERS):
    #     uname = 'f_{0:05d}'.format(i)
    #     train_data['users'].append(uname) 
    #     train_data['user_data'][uname] = {'x': cifa_data_image.tolist(), 'y': cifa_data_label.tolist()}
    #     train_data['num_samples'].append(len(cifa_data_image))

    # #-----------------------------------TEst -------------------------------------#
    # cifa_data_image_test = []
    # cifa_data_label_test = []
    # cifa_data_image_test.extend(testset.data.cpu().detach().numpy())
    # cifa_data_label_test.extend(testset.targets.cpu().detach().numpy())
    # cifa_data_image_test = np.array(cifa_data_image_test)
    # cifa_data_label_test = np.array(cifa_data_label_test)

    # cifa_data = []

    # # Create data structure
    # test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    # for i in range(NUM_USERS):
    #     num_samples = len(cifa_data_image_test)
    #     test_data['users'].append(uname) 
    #     test_data['user_data'][uname] = {'x': cifa_data_image_test.tolist(), 'y': cifa_data_label_test.tolist()}
    #     test_data['num_samples'].append(num_samples)

    return train_data['users'], _ , train_data['user_data'], test_data['user_data']

def read_data(dataset):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    # if(dataset == "Cifar10"):
    #     clients, groups, train_data, test_data = read_cifa_data()
    #     return clients, groups, train_data, test_data

    train_data_dir = os.path.join('data',dataset,'data', 'train')
    test_data_dir = os.path.join('data',dataset,'data', 'test')
    public_data_dir = os.path.join('data', dataset, 'data', 'public')
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    public_data ={}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    public_files = os.listdir(public_data_dir)
    public_files = [f for f in public_files if f.endswith('.json')]
    for f in public_files:
        file_path = os.path.join(public_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        public_data.update(cdata['public_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data, public_data


def read_public_data(data,dataset):
    public_data = data[4]
    X_public, y_public = public_data['x'], public_data['y']
    if (dataset == "Mnist" or dataset == "fmnist"):
        X_public, y_public = public_data['x'], public_data['y']
        X_public = torch.Tensor(X_public).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_public = torch.Tensor(y_public).type(torch.int64)
    elif (dataset == "Cifar10" or dataset =="Cifar100"):

        X_public, y_public = public_data['x'], public_data['y']
        X_public = torch.Tensor(X_public).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_public = torch.Tensor(y_public).type(torch.int64)


    else:
        X_public = torch.Tensor(X_public).type(torch.float32)
        y_public = torch.Tensor(y_public).type(torch.int64)

    public_data = [(x, y) for x, y in zip(X_public, y_public)]
    return public_data

def read_user_data(index,data,dataset):
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    public_data = data[4]
    X_train, y_train, X_test, y_test, X_public, y_public = train_data['x'], train_data['y'], test_data['x'], test_data['y'], public_data['x'], public_data['y']
    if(dataset == "Mnist"):
        X_train, y_train, X_test, y_test, X_public, y_public = train_data['x'], train_data['y'], test_data['x'], test_data['y'], public_data['x'], public_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        X_public = torch.Tensor(X_public).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_public = torch.Tensor(y_public).type(torch.int64)
    elif(dataset == "Cifar10" or dataset == "Cifar100"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        X_public = torch.Tensor(X_public).type(torch.float32)
        y_public = torch.Tensor(y_public).type(torch.int64)
    
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    public_data = [(x, y) for x, y in zip(X_public, y_public)]
    return id, train_data, test_data, public_data

def load_data(data):
    """Loads the dataset of the FL simulation.


    Args:
        config: a map of configurations of the simulation

    Returns:
        A dictionary containing training and testing data for modality A&B and labels.
    """

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
        return (data_train,  data_test, data_public)

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
    public_ratio = 0.11
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
