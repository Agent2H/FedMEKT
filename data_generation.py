import os
import requests
import numpy as np
import pandas as pd
import torch
import datetime

from scipy.io import savemat, loadmat
from scipy.stats import zscore
from FLAlgorithms.trainmodel.ae_model import ResNetMapper

N_DIV_OPP = 100
N_DIV_MHEALTH = 100
N_DIV_URFALL = 10
N_LABEL_DIV_OPP = 15
N_LABEL_DIV_MHEALTH = 9
N_LABEL_DIV_URFALL = 9


def fill_nan(matrix):
    """Fill NaN values with the value of the same column from previous row

    Args:
        matrix: a 2-d numpy matrix
    Return:
        A 2-d numpy matrix with NaN values filled
    """
    m = matrix
    np.nan_to_num(x=m[0, :], copy=False, nan=0.0)
    for row in range(1, m.shape[0]):
        for col in range(m.shape[1]):
            if np.isnan(m[row, col]):
                m[row, col] = m[row-1, col]
    return m


def gen_mhealth(data_path):
    """Generates subjects' data in .mat format from the mHealth dataset.

    The experiments on the mHealth dataset are done in the fashion of leave-one-subject-off.
    So the .mat data is indexed by subjects instead of "training", "validating", and "testing".

    Args:
        data_path: the path of the mHealth dataset.

    Returns:
        None
    """
    acce_columns = [0, 1, 2, 5, 6, 7, 14, 15, 16]
    gyro_columns = [8, 9, 10, 17, 18, 19]
    mage_columns = [11, 12, 13, 20, 21, 22]
    y_column = 23
    mdic = {}
    labels = set()
    shape_list = []
    for i in range(1, 11):
        s_data = np.loadtxt(os.path.join(
            data_path, "mhealth", f"mHealth_subject{i}.log"))
        x_acce = fill_nan(s_data[:, acce_columns])
        x_gyro = fill_nan(s_data[:, gyro_columns])
        x_mage = fill_nan(s_data[:, mage_columns])
        y = s_data[:, y_column]
        mdic[f"s{i}_acce"] = x_acce
        mdic[f"s{i}_gyro"] = x_gyro
        mdic[f"s{i}_mage"] = x_mage
        mdic[f"s{i}_y"] = y
        labels = labels.union(set(y))
        print(f"shape of participant {i}: {s_data.shape}")
        shape_list.append(s_data.shape[0])

    print(f"mean:{np.mean(shape_list)}, std:{np.std(shape_list)}")
    unique_y = list(labels)
    unique_y.sort()

    y_map = {}
    for idx, y in enumerate(unique_y):
        y_map[y] = idx
    for i in range(1, 11):
        mdic[f"s{i}_y"] = np.squeeze(
            np.vectorize(y_map.get)(mdic[f"s{i}_y"]))
    savemat(os.path.join(data_path, "mhealth", "mhealth.mat"), mdic)


def gen_opp(data_path):
    """Generates training, validating, and testing data from Opp datasets

    Args:
        data_path: the path of the Opportunity challenge dataset

    Returns:
        None
    """
    acce_columns = [i-1 for i in range(2, 41)]
    acce_columns.extend([46, 47, 48, 55, 56, 57, 64, 65, 66, 73, 74,
                         75, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105, 106, ])
    gyro_columns = [40, 41, 42, 49, 50, 51,
                    58, 59, 60, 67, 68, 69, 66, 67, 68, ]
    # Loads the run 2 from subject 1 as validating data
    data_valid = np.loadtxt(os.path.join(data_path, "opp", "S1-ADL2.dat"))
    x_valid_acce = fill_nan(data_valid[:, acce_columns])
    x_valid_gyro = fill_nan(data_valid[:, gyro_columns])
    y_valid = data_valid[:, 115]

    # Loads the runs 4 and 5 from subjects 2 and 3 as testing data
    runs_test = []
    idxs_test = []
    for r in [4, 5]:
        for s in [2, 3]:
            runs_test.append(np.loadtxt(os.path.join(
                data_path, "opp", f"S{s}-ADL{r}.dat")))
            idxs_test.append((r, s))
    data_test = np.concatenate(runs_test)
    x_test_acce = fill_nan(data_test[:, acce_columns])
    x_test_gyro = fill_nan(data_test[:, gyro_columns])
    y_test = data_test[:, 115]

    #create public data for opp
    runs_public = []
    idxs_public = []
    for r in [4, 5]:
        for s in [1, 4]:
            runs_public.append(np.loadtxt(os.path.join(
                data_path, "opp", f"S{s}-ADL{r}.dat")))
            idxs_public.append((r, s))
    data_public = np.concatenate(runs_public)
    x_public_acce = fill_nan(data_public[:, acce_columns])
    x_public_gyro = fill_nan(data_public[:, gyro_columns])
    y_public = data_public[:, 115]

    # Loads the remaining runs as training data
    runs_train = []
    for r in range(1, 6):
        for s in range(1, 5):
            if (r, s) not in idxs_test and (r,s) not in idxs_public:
                runs_train.append(np.loadtxt(os.path.join(
                    data_path, "opp", f"S{s}-ADL{r}.dat")))
    data_train = np.concatenate(runs_train)
    x_train_acce = fill_nan(data_train[:, acce_columns])
    x_train_gyro = fill_nan(data_train[:, gyro_columns])
    y_train = data_train[:, 115]

    # Changes labels to (0, 1, ...)
    unique_y = list(set(y_train).union(set(y_valid)).union(set(y_test)).union(set(y_public)))
    unique_y.sort()

    y_map = {}
    for idx, y in enumerate(unique_y):
        y_map[y] = idx
    y_train = np.vectorize(y_map.get)(y_train)
    y_valid = np.vectorize(y_map.get)(y_valid)
    y_test = np.vectorize(y_map.get)(y_test)
    y_public = np.vectorize(y_map.get)(y_public)

    mdic = {}
    mdic["x_train_acce"] = x_train_acce
    mdic["x_train_gyro"] = x_train_gyro
    mdic["y_train"] = np.squeeze(y_train)
    mdic["x_valid_acce"] = x_valid_acce
    mdic["x_valid_gyro"] = x_valid_gyro
    mdic["y_valid"] = np.squeeze(y_valid)  # This only has 17 classes
    mdic["x_test_acce"] = x_test_acce
    mdic["x_test_gyro"] = x_test_gyro
    mdic["y_test"] = np.squeeze(y_test)
    mdic["x_public_acce"] = x_public_acce
    mdic["x_public_gyro"] = x_public_gyro
    mdic["y_public"] = np.squeeze(y_public)

    savemat(os.path.join(data_path, "opp", "opp.mat"), mdic)


def gen_ur_fall(data_path):
    """Generates training and testing data for UR Fall datasets.

    Args:
        data_path: the path of the UR Fall datasets.

    Returns:
        None
    """
    # headers
    # fall (0 or 1), run (1-40 for fall=0, 1-30 for fall=1), frame, HeightWidthRatio, MajorMinorRatio, BoundingBoxOccupancy, MaxStdXZ, HHmaxRatio, H, D, P40, acce_x, acce_y, acce_z, y
    a_list = []
    runs = [40, 30]
    shape_list = []
    for fall in range(2):
        prefix = "fall" if fall == 1 else "adl"
        f_labelled = os.path.join(
            data_path, "ur_fall", prefix, f"urfall-features-cam0-{prefix}s.csv")
        df_labelled = pd.read_csv(
            f_labelled, delimiter=",", header=None, usecols=list(range(11)))
        for run in range(1, runs[fall]+1):
            f_acc = os.path.join(data_path, "ur_fall", prefix,
                                 "acc", f"{prefix}-{str(run).zfill(2)}-acc.csv")
            f_sync = os.path.join(
                data_path, "ur_fall", prefix, "sync", f"{prefix}-{str(run).zfill(2)}-data.csv")
            data_acce = np.genfromtxt(f_acc, delimiter=",")
            data_sync = np.genfromtxt(f_sync, delimiter=",")
            df_label_part = df_labelled[df_labelled[0]
                                        == f"{prefix}-{str(run).zfill(2)}"]
            n_rows = df_label_part.shape[0]
            a = np.zeros([n_rows, 15])
            a[:, 0] = fall
            a[:, 1] = run
            a[:, 2] = df_label_part[1].to_numpy()
            a[:, 3:11] = df_label_part[df_label_part.columns.intersection(
                list(range(3, 11)))].to_numpy()
            a[:, 14] = df_label_part[2].to_numpy()
            mask = [x in a[:, 2] for x in data_sync[:, 0]]
            timestamps = data_sync[mask, 1]
            acce_xyz = np.empty((0, 3), dtype=np.float64)
            row_acce_data = 0
            for ts in timestamps:
                while row_acce_data < data_acce.shape[0] and data_acce[row_acce_data, 0] < ts:
                    row_acce_data += 1
                if row_acce_data >= data_acce.shape[0]:
                    break
                if abs(data_acce[row_acce_data, 0] - ts) < abs(data_acce[row_acce_data-1, 0] - ts):
                    acce_xyz = np.append(
                        acce_xyz, [data_acce[row_acce_data, 2:5]], axis=0)
                else:
                    acce_xyz = np.append(
                        acce_xyz, [data_acce[row_acce_data-1, 2:5]], axis=0)
                acce_xyz = [np.append(acce_xyz, [data_acce[row_acce_data, 2:5]], axis=0) if abs(data_acce[row_acce_data, 0] - ts) < abs(data_acce[row_acce_data - 1, 0] - ts) else
                            np.append(acce_xyz, [data_acce[row_acce_data - 1, 2:5]], axis=0)]
            if acce_xyz.shape[0] < a.shape[0]:
                n = a.shape[0] - acce_xyz.shape[0]
                a = a[:-n, :]
            a[:, 11:14] = acce_xyz
            a_list.append(a)
            shape_list.append(a.shape[0])
            print(f"shape: {a.shape}")
    print(f"mean:{np.mean(shape_list)}, std:{np.std(shape_list)}")

    data = np.concatenate(a_list)
    mdic = {}
    mdic["depth"] = data[:, 0:11]
    mdic["acce"] = data[:, [0, 1, 2, 11, 12, 13]]
    mdic["y"] = data[:, [0, 1, 2, 14]]
    idxs_rgb = data[:, [0, 1, 2]]
    rgb_features = ResNetMapper.map(idxs_rgb).numpy()
    mdic["rgb"] = np.empty((data.shape[0], rgb_features.shape[1]+3))
    mdic["rgb"][:, [0, 1, 2]] = idxs_rgb
    mdic["rgb"][:, range(3, rgb_features.shape[1]+3)] = rgb_features

    y_old = data[:, 14]
    unique_y = list(set(y_old))
    unique_y.sort()

    y_map = {}
    for idx, y in enumerate(unique_y):
        y_map[y] = idx
    mdic["y"][:, 3] = np.vectorize(y_map.get)(y_old)

    savemat(os.path.join(data_path, "ur_fall", "ur_fall.mat"), mdic)
    print("finish")

if __name__ == "__main__":
    gen_opp("data")
    # gen_mhealth("data")
    # download_UR_fall()
    # gen_ur_fall("data")
    pass