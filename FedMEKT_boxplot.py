# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:19:14 2017

@author: Minh
"""
import os

import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

Table_index1 = 90
Table_index2 = 100
Start_index = 80
def plot_final():
    df_iters = read_files_opp_global()  # 3 algorithms
    df_iters = read_files_opp_multimodal_unimodal()
#     plot_box(df_iters,1)
#     plt.savefig("figs/mnist_fixed_users_boxplot.pdf")
    df_iters = read_files1_global_mhealth()
    df_iters = read_files1_multimodal_unimodal_mhealth()
#     plot_box(df_iters,2)
#     plt.savefig("figs/mnist_subset_users_boxplot.pdf")
#     df_iters = read_files_global_urfall()
    df_iters = read_files_UR_FALL_multimodal_full()
    df_iters= read_files_UR_FALL_multimodal_unimodal_full()
    df_iters =read_files_UR_FALL_proxy_data()
    df_iters =read_files_UR_FALL_R_steps()
    df_iters =read_files_UR_FALL_local_epoch()
    df_iters = read_files_UR_FALL_ablation()
#     plot_box(df_iters,3)
#     plt.savefig("figs/fmnist_fixed_users_boxplot.pdf")
    df_iters = read_files3()
#     plot_box(df_iters,4)
#     plt.savefig("figs/fmnist_subset_users_boxplot.pdf")
    df_iters = read_files4()
#     plot_box(df_iters, 5)
#     plt.savefig("figs/cifar10_fixed_users_boxplot.pdf")
#     df_iters = read_files5()
#     plot_box(df_iters, 6)
#     plt.savefig("figs/cifar10_subset_users_boxplot.pdf")
#     df_iters = read_files6()
#     plot_box(df_iters, 7)
#     plt.savefig("figs/cifar100_fixed_users_boxplot.pdf")
#     df_iters = read_files7()
#     plot_box(df_iters, 8)
#     plt.savefig("figs/cifar100_subset_users_boxplot.pdf")
#     # plt.ylim(0, 100)
#     # plt.savefig("figs/mnist_fixed_users_boxplot.png")
#     # plt.savefig("figs/mnist_subset_users_boxplot.png")
#     # plt.savefig("figs/fmnist_fixed_users_boxplot.png")
#     # plt.savefig("figs/fmnist_subset_users_boxplot.png")
#     plt.show()
# def plot_box(df_iters,figure_index):
#
#
#     # plt.figure(2, figsize=(7., 5.1))
#     # plt.figure(2, figsize=(8.7, 5.8))
#     plt.figure(figure_index, figsize=(35, 20))
#     # plt.figure(2,figsize=(4.5,5))
#     # sns.set_style("whitegrid")
#     sns.set_context("notebook", font_scale=3.3)
#     sns.swarmplot(x="Algorithm", y="Accuracy", data=df_iters)
#     plt.tight_layout(pad=3, w_pad=3, h_pad=3)
#     sns.boxplot(x="Algorithm", y="Accuracy", data=df_iters, showcaps=True, boxprops={'facecolor': 'None'},
#                 showfliers=False, whiskerprops={'linewidth': 2}, linewidth=2)
#     plt.xlabel('Algorithm', fontsize=39)
#     plt.ylabel('Testing Accuracy', fontsize=39)
#     # plt.ylim(0, 100)
#     # plt.savefig("figs/mnist_fixed_users_boxplot.png")
#     # plt.savefig("figs/mnist_subset_users_boxplot.png")
#     # plt.savefig("figs/fmnist_fixed_users_boxplot.png")
#     # plt.savefig("figs/fmnist_subset_users_boxplot.png")


def read_files_opp_global():

    filename = 'results_fig'+ 'CDKT_mnist_I100_sTrue_fFalse_RFFalse_SSFalse_gmKL_lmNorm2.h5'
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'mmFedAvg_opp_LAB_TA_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'mmFedAvg_opp_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'mmFedProx_opp_LAB_TA_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'mmFedProx_opp_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'mmMOON_opp_LAB_TA_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory,'mmMOON_opp_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df7 = h5py.File(os.path.join(directory, 'FedMEKT_opp_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df8 = h5py.File(os.path.join(directory,'FedMEKT_opp_LAB_TB_I100_Maacce_Mbgyro_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df9 = h5py.File(os.path.join(directory,'FedMEFKT_opp_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.009_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df10 = h5py.File(os.path.join(directory, 'FedMEFKT_opp_LAB_TB_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    #
    # test1= df8['root_test'][Table_index1:Table_index2]
    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    # print("spec test", np.median(df8['avg_local_f1_acc'][Table_index1:Table_index2]) )
    # glob5 = df5['root_test'][Table_index1:Table_index2]
    # glob6 = df6['root_test'][Table_index1:Table_index2]



    print ("---------------  OPP RESULTS --------------")

    print("fedavg test acce glob:", np.median(glob1))
    print("fedavg test gyro glob:", np.median(glob2))
    print("mmFedProx test acce glob:", np.median(glob3))
    print("mmFedProx test gyro glob:", np.median(glob4))
    print("mmMOON test acce glob:", np.median(glob5))
    print("mmMOON test gyro glob:", np.median(glob6))
    print("FedMEKT test acce glob:", np.median(glob7))
    print("FedMEKT test gyro glob:", np.median(glob8))
    print("FedMEFKT test acce glob:", np.median(glob9))
    print("FedMEFKT test gyro glob:", np.median(glob10))

    #
    # print("fedekt con test gyro glob:", np.median(glob5))
    #
    # print("fedekt con test acce glob:", np.median(glob6))

    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    # data = np.concatenate((glob1[np.newaxis,:],glob2[np.newaxis,:],glob3[np.newaxis,:],glob4[np.newaxis,:],gen1[np.newaxis,:],gen2[np.newaxis,:],gen3[np.newaxis,:],gen4[np.newaxis,:]), axis = 0)
    # # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # # data = np.concatenate((stop1, stop3), axis=1)
    # # iters_cols =['Centralized','Decentralized']
    # df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    # df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # # print(df_iters)
    # return df_iters

def read_files_opp_multimodal_unimodal():

    filename = 'results_fig'+ 'CDKT_mnist_I100_sTrue_fFalse_RFFalse_SSFalse_gmKL_lmNorm2.h5'
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'mmFedAvg_opp_LAB_TA_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'mmFedAvg_opp_LAB_TB_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'mmFedProx_opp_LAB_TA_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'mmFedProx_opp_LAB_TB_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'mmMOON_opp_LAB_TA_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory,'mmMOON_opp_LAB_TB_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df7 = h5py.File(os.path.join(directory, 'FedMEKT_opp_LAB_TA_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.009_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df8 = h5py.File(os.path.join(directory,'FedMEKT_opp_LAB_TB_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df9 = h5py.File(os.path.join(directory,'FedMEFKT_opp_LAB_TA_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.009_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df10 = h5py.File(os.path.join(directory, 'FedMEFKT_opp_LAB_TB_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]

    #
    # test1= df8['root_test'][Table_index1:Table_index2]
    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    # print("spec test", np.median(df8['avg_local_f1_acc'][Table_index1:Table_index2]) )
    # glob5 = df5['root_test'][Table_index1:Table_index2]
    # glob6 = df6['root_test'][Table_index1:Table_index2]



    print ("---------------  OPP Multimodal+Unimodal RESULTS --------------")

    print("fedavg test acce glob:", np.median(glob1))
    print("fedavg test gyro glob:", np.median(glob2))
    print("mmFedProx test acce glob:", np.median(glob3))
    print("mmFedProx test gyro glob:", np.median(glob4))
    print("mmMOON test acce glob:", np.median(glob5))
    print("mmMOON test gyro glob:", np.median(glob6))
    print("FedMEKT test acce glob:", np.median(glob7))
    print("FedMEKT test gyro glob:", np.median(glob8))
    print("FedMEFKT test acce glob:", np.median(glob9))
    print("FedMEFKT test gyro glob:", np.median(glob10))

    #
    # print("fedekt con test gyro glob:", np.median(glob5))
    #
    # print("fedekt con test acce glob:", np.median(glob6))

    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    # data = np.concatenate((glob1[np.newaxis,:],glob2[np.newaxis,:],glob3[np.newaxis,:],glob4[np.newaxis,:],gen1[np.newaxis,:],gen2[np.newaxis,:],gen3[np.newaxis,:],gen4[np.newaxis,:]), axis = 0)
    # # print(data.transpose())
    # iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
    #               '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # # data = np.concatenate((stop1, stop3), axis=1)
    # # iters_cols =['Centralized','Decentralized']
    # df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    # df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # # print(df_iters)
    # return df_iters

def read_files1_global_mhealth():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.03_eta0.03_beta0.006_gamma0.006_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.09_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df7 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.03_eta0.03_beta0.006_gamma0.006_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df8 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df9 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),   'r')
    df10 = h5py.File(os.path.join(directory,'mmFedProx_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df11 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.06_gamma0.06_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df12 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')

    df13 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.03_eta0.03_beta0.006_gamma0.006_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df14 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df15 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df16 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df17 = h5py.File(os.path.join(directory, 'mmMOON_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.06_gamma0.06_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df18 = h5py.File(os.path.join(directory, 'mmMOON_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df19 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df20 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.03_eta0.03_beta0.006_gamma0.006_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df21 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df22 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df23 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df24 = h5py.File(os.path.join(directory, 'FedMEKT_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.06_gamma0.06_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df25 = h5py.File(os.path.join(directory, 'FedMEFKT_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df26 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df27 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df28 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df29 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TA_A0_B0_AB30_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df30 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.05_eta0.05_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')

    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    glob11 = df11['root_test'][Table_index1:Table_index2]
    glob12 = df12['root_test'][Table_index1:Table_index2]
    glob13 = df13['root_test'][Table_index1:Table_index2]
    glob14 = df14['root_test'][Table_index1:Table_index2]
    glob15 = df15['root_test'][Table_index1:Table_index2]
    glob16 = df16['root_test'][Table_index1:Table_index2]
    glob17 = df17['root_test'][Table_index1:Table_index2]
    glob18 = df18['root_test'][Table_index1:Table_index2]
    glob19 = df19['root_test'][Table_index1:Table_index2]
    glob20 = df20['root_test'][Table_index1:Table_index2]
    glob21 = df21['root_test'][Table_index1:Table_index2]
    glob22 = df22['root_test'][Table_index1:Table_index2]
    glob23 = df23['root_test'][Table_index1:Table_index2]
    glob24 = df24['root_test'][Table_index1:Table_index2]
    glob25 = df25['root_test'][Table_index1:Table_index2]
    glob26 = df26['root_test'][Table_index1:Table_index2]
    glob27 = df27['root_test'][Table_index1:Table_index2]
    glob28 = df28['root_test'][Table_index1:Table_index2]
    glob29 = df29['root_test'][Table_index1:Table_index2]
    glob30 = df30['root_test'][Table_index1:Table_index2]

    print("--------------- SUBSET mHealth RESULTS --------------")

    print("---------------ACCE-GYRO --------------")
    print("mmFedAvg acce glob:", np.median(glob1))
    print("mmFedAvg gyro glob:", np.median(glob2))

    print("mmFedProx acce glob:", np.median(glob7))
    print("mmFedProx gyro glob:", np.median(glob8))

    print("mmMOON acce glob:", np.median(glob13))
    print("mmMOON gyro glob:", np.median(glob14))

    print("FedMEKT acce glob:", np.median(glob19))
    print("FedMEKT gyro glob:", np.median(glob20))

    print("FedMEFKT acce glob:", np.median(glob25))
    print("FedMEFKT gyro glob:", np.median(glob26))


    print("---------------ACCE-MAGE --------------")

    print("mmFedAvg acce glob:", np.median(glob3))

    print("mmFedAvg mage glob:", np.median(glob4))


    print("mmFedProx acce glob:", np.median(glob9))
    print("mmFedProx mage glob:", np.median(glob10))

    print("mmMOON acce glob:", np.median(glob15))
    print("mmMOON mage glob:", np.median(glob16))

    print("FedMEKT acce glob:", np.median(glob21))
    print("FedMEKT mage glob:", np.median(glob22))

    print("FedMEFKT acce glob:", np.median(glob27))
    print("FedMEFKT mage glob:", np.median(glob28))


    print("---------------GYRO-MAGE --------------")
    print("mmFedAvg gyro glob:", np.median(glob5))
    print("mmFedAvg mage glob:", np.median(glob6))


    print("mmFedProx gyro glob:", np.median(glob11))
    print("mmFedProx mage glob:", np.median(glob12))
    print("mmMOON gyro glob:", np.median(glob17))
    print("mmMOON mage glob:", np.median(glob18))
    print("FedMEKT gyro glob:", np.median(glob23))
    print("FedMEKT mage glob:", np.median(glob24))
    print("FedMEFKT gyro glob:", np.median(glob29))
    print("FedMEFKT mage glob:", np.median(glob30))


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                           glob3[np.newaxis, :],  glob4[np.newaxis, :],glob5[np.newaxis, :], glob6[np.newaxis, :]), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
                  '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters

def read_files1_multimodal_unimodal_mhealth():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TA_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TB_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TA_A10_B10_AB10_I100_Maacce_Mbmage_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TB_A10_B10_AB10_I100_Maacce_Mbmage_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TA_A10_B10_AB10_I100_Magyro_Mbmage_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TB_A10_B10_AB10_I100_Magyro_Mbmage_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df7 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TA_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df8 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TB_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df9 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TA_A10_B10_AB10_I100_Maacce_Mbmage_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),   'r')
    df10 = h5py.File(os.path.join(directory,'mmFedProx_mhealth_LAB_TB_A10_B10_AB10_I100_Maacce_Mbmage_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df11 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TA_A10_B10_AB10_I100_Magyro_Mbmage_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df12 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TB_A10_B10_AB10_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')

    df13 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TA_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df14 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TB_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df15 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TA_A10_B10_AB10_I100_Maacce_Mbmage_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df16 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TB_A10_B10_AB10_I100_Maacce_Mbmage_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df17 = h5py.File(os.path.join(directory, 'mmMOON_mhealth_LAB_TA_A10_B10_AB10_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df18 = h5py.File(os.path.join(directory, 'mmMOON_mhealth_LAB_TB_A10_B10_AB10_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df19 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TA_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df20 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TB_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.01_eta0.01_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df21 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TA_A10_B10_AB10_I100_Maacce_Mbmage_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df22 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TB_A10_B10_AB10_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df23 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TA_A10_B10_AB10_I100_Magyro_Mbmage_alpha0.06_eta0.06_beta0.06_gamma0.06_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df24 = h5py.File(os.path.join(directory, 'FedMEKT_mhealth_LAB_TB_A10_B10_AB10_I100_Magyro_Mbmage_alpha0.05_eta0.05_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df25 = h5py.File(os.path.join(directory, 'FedMEFKT_mhealth_LAB_TA_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.0_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df26 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TB_A10_B10_AB10_I100_Maacce_Mbgyro_alpha0.01_eta0.01_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df27 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TA_A10_B10_AB10_I100_Maacce_Mbmage_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df28 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TB_A10_B10_AB10_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df29 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TA_A10_B10_AB10_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df30 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TB_A10_B10_AB10_I100_Magyro_Mbmage_alpha0.05_eta0.05_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')

    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    glob11 = df11['root_test'][Table_index1:Table_index2]
    glob12 = df12['root_test'][Table_index1:Table_index2]
    glob13 = df13['root_test'][Table_index1:Table_index2]
    glob14 = df14['root_test'][Table_index1:Table_index2]
    glob15 = df15['root_test'][Table_index1:Table_index2]
    glob16 = df16['root_test'][Table_index1:Table_index2]
    glob17 = df17['root_test'][Table_index1:Table_index2]
    glob18 = df18['root_test'][Table_index1:Table_index2]
    glob19 = df19['root_test'][Table_index1:Table_index2]
    glob20 = df20['root_test'][Table_index1:Table_index2]
    glob21 = df21['root_test'][Table_index1:Table_index2]
    glob22 = df22['root_test'][Table_index1:Table_index2]
    glob23 = df23['root_test'][Table_index1:Table_index2]
    glob24 = df24['root_test'][Table_index1:Table_index2]
    glob25 = df25['root_test'][Table_index1:Table_index2]
    glob26 = df26['root_test'][Table_index1:Table_index2]
    glob27 = df27['root_test'][Table_index1:Table_index2]
    glob28 = df28['root_test'][Table_index1:Table_index2]
    glob29 = df29['root_test'][Table_index1:Table_index2]
    glob30 = df30['root_test'][Table_index1:Table_index2]

    print("--------------- SUBSET mHealth Multimodal+Unimodal RESULTS --------------")

    print("---------------ACCE-GYRO --------------")
    print("mmFedAvg acce glob:", np.median(glob1))
    print("mmFedAvg gyro glob:", np.median(glob2))

    print("mmFedProx acce glob:", np.median(glob7))
    print("mmFedProx gyro glob:", np.median(glob8))

    print("mmMOON acce glob:", np.median(glob13))
    print("mmMOON gyro glob:", np.median(glob14))

    print("FedMEKT acce glob:", np.median(glob19))
    print("FedMEKT gyro glob:", np.median(glob20))

    print("FedMEFKT acce glob:", np.median(glob25))
    print("FedMEFKT gyro glob:", np.median(glob26))


    print("---------------ACCE-MAGE --------------")

    print("mmFedAvg acce glob:", np.median(glob3))

    print("mmFedAvg mage glob:", np.median(glob4))


    print("mmFedProx acce glob:", np.median(glob9))
    print("mmFedProx mage glob:", np.median(glob10))

    print("mmMOON acce glob:", np.median(glob15))
    print("mmMOON mage glob:", np.median(glob16))

    print("FedMEKT acce glob:", np.median(glob21))
    print("FedMEKT mage glob:", np.median(glob22))

    print("FedMEFKT acce glob:", np.median(glob27))
    print("FedMEFKT mage glob:", np.median(glob28))


    print("---------------GYRO-MAGE --------------")
    print("mmFedAvg gyro glob:", np.median(glob5))
    print("mmFedAvg mage glob:", np.median(glob6))


    print("mmFedProx gyro glob:", np.median(glob11))
    print("mmFedProx mage glob:", np.median(glob12))
    print("mmMOON gyro glob:", np.median(glob17))
    print("mmMOON mage glob:", np.median(glob18))
    print("FedMEKT gyro glob:", np.median(glob23))
    print("FedMEKT mage glob:", np.median(glob24))
    print("FedMEFKT gyro glob:", np.median(glob29))
    print("FedMEFKT mage glob:", np.median(glob30))


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                           glob3[np.newaxis, :],  glob4[np.newaxis, :],glob5[np.newaxis, :], glob6[np.newaxis, :]), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
                  '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters

def global_mhealth_multimodal_unimodal():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.03_eta0.03_beta0.006_gamma0.006_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.09_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'mmFedAvg_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df7 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.03_eta0.03_beta0.006_gamma0.006_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df8 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df9 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),   'r')
    df10 = h5py.File(os.path.join(directory,'mmFedProx_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df11 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.06_gamma0.06_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df12 = h5py.File(os.path.join(directory, 'mmFedProx_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')

    df13 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.03_eta0.03_beta0.006_gamma0.006_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df14 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df15 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df16 = h5py.File(os.path.join(directory,'mmMOON_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df17 = h5py.File(os.path.join(directory, 'mmMOON_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.06_gamma0.06_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df18 = h5py.File(os.path.join(directory, 'mmMOON_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df19 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df20 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.03_eta0.03_beta0.006_gamma0.006_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df21 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df22 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df23 = h5py.File(os.path.join(directory,'FedMEKT_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df24 = h5py.File(os.path.join(directory, 'FedMEKT_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.06_gamma0.06_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df25 = h5py.File(os.path.join(directory, 'FedMEFKT_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df26 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df27 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df28 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df29 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TA_A0_B0_AB30_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df30 = h5py.File(os.path.join(directory,'FedMEFKT_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.05_eta0.05_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')

    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    glob11 = df11['root_test'][Table_index1:Table_index2]
    glob12 = df12['root_test'][Table_index1:Table_index2]
    glob13 = df13['root_test'][Table_index1:Table_index2]
    glob14 = df14['root_test'][Table_index1:Table_index2]
    glob15 = df15['root_test'][Table_index1:Table_index2]
    glob16 = df16['root_test'][Table_index1:Table_index2]
    glob17 = df17['root_test'][Table_index1:Table_index2]
    glob18 = df18['root_test'][Table_index1:Table_index2]
    glob19 = df19['root_test'][Table_index1:Table_index2]
    glob20 = df20['root_test'][Table_index1:Table_index2]
    glob21 = df21['root_test'][Table_index1:Table_index2]
    glob22 = df22['root_test'][Table_index1:Table_index2]
    glob23 = df23['root_test'][Table_index1:Table_index2]
    glob24 = df24['root_test'][Table_index1:Table_index2]
    glob25 = df25['root_test'][Table_index1:Table_index2]
    glob26 = df26['root_test'][Table_index1:Table_index2]
    glob27 = df27['root_test'][Table_index1:Table_index2]
    glob28 = df28['root_test'][Table_index1:Table_index2]
    glob29 = df29['root_test'][Table_index1:Table_index2]
    glob30 = df30['root_test'][Table_index1:Table_index2]

    print("--------------- SUBSET mHealth RESULTS --------------")

    print("---------------ACCE-GYRO --------------")
    print("mmFedAvg acce glob:", np.median(glob1))
    print("mmFedAvg gyro glob:", np.median(glob2))

    print("mmFedProx acce glob:", np.median(glob7))
    print("mmFedProx gyro glob:", np.median(glob8))

    print("mmMOON acce glob:", np.median(glob13))
    print("mmMOON gyro glob:", np.median(glob14))

    print("FedMEKT acce glob:", np.median(glob19))
    print("FedMEKT gyro glob:", np.median(glob20))

    print("FedMEFKT acce glob:", np.median(glob25))
    print("FedMEFKT gyro glob:", np.median(glob26))


    print("---------------ACCE-MAGE --------------")

    print("mmFedAvg acce glob:", np.median(glob3))

    print("mmFedAvg mage glob:", np.median(glob4))


    print("mmFedProx acce glob:", np.median(glob9))
    print("mmFedProx mage glob:", np.median(glob10))

    print("mmMOON acce glob:", np.median(glob15))
    print("mmMOON mage glob:", np.median(glob16))

    print("FedMEKT acce glob:", np.median(glob21))
    print("FedMEKT mage glob:", np.median(glob22))

    print("FedMEFKT acce glob:", np.median(glob27))
    print("FedMEFKT mage glob:", np.median(glob28))


    print("---------------GYRO-MAGE --------------")
    print("mmFedAvg gyro glob:", np.median(glob5))
    print("mmFedAvg mage glob:", np.median(glob6))


    print("mmFedProx gyro glob:", np.median(glob11))
    print("mmFedProx mage glob:", np.median(glob12))
    print("mmMOON gyro glob:", np.median(glob17))
    print("mmMOON mage glob:", np.median(glob18))
    print("FedMEKT gyro glob:", np.median(glob23))
    print("FedMEKT mage glob:", np.median(glob24))
    print("FedMEFKT gyro glob:", np.median(glob29))
    print("FedMEFKT mage glob:", np.median(glob30))


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                           glob3[np.newaxis, :],  glob4[np.newaxis, :],glob5[np.newaxis, :], glob6[np.newaxis, :]), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
                  '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters

def read_files_global_urfall():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'mmFedAvg_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5'),  'r')
    df2 = h5py.File(os.path.join(directory,'mmFedAvg_ur_fall_LAB_TB_I100_Maacce_Mbrgb_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsFalse.h5'), 'r')
    df3 = h5py.File(os.path.join(directory,'mmFedAvg_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory,'mmFedAvg_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'mmFedAvg_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.05_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5'),'r')
    df6 = h5py.File(os.path.join(directory,'mmFedAvg_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.05_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5'), 'r')
    df7 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse.h5'), 'r')
    df8 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbrgb_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsFalse.h5'), 'r')
    df9 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse.h5'),'r')
    df10 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse.h5'),'r')
    df11 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.1_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse.h5'),'r')
    df12 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5'),'r')
    df13 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5'), 'r')
    df14 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5'), 'r')
    df15 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.1_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5'),'r')
    df16 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5'), 'r')
    df17 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5'), 'r')
    df18 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.07_eta0.07_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5'), 'r')
    df19 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5'),'r')
    df20 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5'), 'r')
    df21 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5'),'r')
    df22 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5'),'r')
    df23 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbrgb_alpha0.04_eta0.04_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5'),'r')
    df24 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbrgb_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5'), 'r')
    df25 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse.h5'),'r')
    df26 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse.h5'), 'r')
    df27 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5'), 'r')
    df28 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5'), 'r')
    df29 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse.h5'),'r')
    df30 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse.h5'),'r')
    df31 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse.h5'),'r')
    df32 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse.h5'),'r')
    df33 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.1_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse.h5'),'r')
    df34 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.1_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse.h5'), 'r')
    df35 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse.h5'), 'r')
    df36 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse.h5'),'r')

    df37 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'),'r')
    df38 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'),'r')
    df39 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5'),'r')
    df40 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5'),'r')
    df41 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'), 'r')
    df42 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'),'r')
    df43 = h5py.File(os.path.join(directory,  'mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'),'r')
    df44 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'),'r')
    df45 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.1_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'), 'r')
    df46 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.1_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'), 'r')
    df47 = h5py.File(os.path.join(directory, 'mmFedEKT_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'), 'r')
    df48 = h5py.File(os.path.join(directory,'mmFedEKT_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'),'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    spec1 = df['avg_local_f1_acc'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    spec2 = df2['avg_local_f1_acc'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    spec3 = df3['avg_local_f1_acc'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    spec4 = df4['avg_local_f1_acc'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    spec5 = df5['avg_local_f1_acc'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    spec6 = df6['avg_local_f1_acc'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    spec7 = df7['avg_local_f1_acc'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    spec8 = df8['avg_local_f1_acc'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    spec9 = df9['avg_local_f1_acc'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    spec10= df10['avg_local_f1_acc'][Table_index1:Table_index2]
    glob11 = df11['root_test'][Table_index1:Table_index2]
    spec11 = df11['avg_local_f1_acc'][Table_index1:Table_index2]
    glob12 = df12['root_test'][Table_index1:Table_index2]
    spec12 = df12['avg_local_f1_acc'][Table_index1:Table_index2]
    spec13 = df13['avg_local_f1_acc'][Table_index1:Table_index2]
    spec14 = df14['avg_local_f1_acc'][Table_index1:Table_index2]
    spec15 = df15['avg_local_f1_acc'][Table_index1:Table_index2]
    spec16 = df16['avg_local_f1_acc'][Table_index1:Table_index2]
    glob17 = df17['root_test'][Table_index1:Table_index2]
    glob18 = df18['root_test'][Table_index1:Table_index2]
    glob19 = df19['root_test'][Table_index1:Table_index2]
    glob20 = df20['root_test'][Table_index1:Table_index2]
    glob21 = df21['root_test'][Table_index1:Table_index2]
    spec22 = df22['avg_local_f1_acc'][Table_index1:Table_index2]
    glob23 = df23['root_test'][Table_index1:Table_index2]
    spec24 = df24['avg_local_f1_acc'][Table_index1:Table_index2]
    glob25 = df25['root_test'][Table_index1:Table_index2]
    glob26 = df26['root_test'][Table_index1:Table_index2]
    glob27 = df27['root_test'][Table_index1:Table_index2]
    glob28 = df28['root_test'][Table_index1:Table_index2]
    glob29 = df29['root_test'][Table_index1:Table_index2]
    glob30 = df30['root_test'][Table_index1:Table_index2]
    glob31 = df31['root_test'][Table_index1:Table_index2]
    glob32 = df32['root_test'][Table_index1:Table_index2]
    glob33 = df33['root_test'][Table_index1:Table_index2]
    glob34 = df34['root_test'][Table_index1:Table_index2]
    glob35 = df35['root_test'][Table_index1:Table_index2]
    glob36 = df36['root_test'][Table_index1:Table_index2]

    glob37 = df37['root_test'][Table_index1:Table_index2]
    glob38 = df38['root_test'][Table_index1:Table_index2]
    glob39 = df39['root_test'][Table_index1:Table_index2]
    glob40 = df40['root_test'][Table_index1:Table_index2]
    glob41 = df41['root_test'][Table_index1:Table_index2]
    glob42 = df42['root_test'][Table_index1:Table_index2]
    glob43 = df43['root_test'][Table_index1:Table_index2]
    glob44 = df44['root_test'][Table_index1:Table_index2]
    glob45 = df45['root_test'][Table_index1:Table_index2]
    glob46 = df46['root_test'][Table_index1:Table_index2]
    glob47 = df47['root_test'][Table_index1:Table_index2]
    glob48 = df48['root_test'][Table_index1:Table_index2]
    print("--------------- SUBSET UR FALL RESULTS --------------")

    print("---------------ACCE-RGB --------------")
    print("mmFedAvg acce glob:", np.median(glob1))
    print("mmFedAvg acce local:", np.median(spec1))
    print("mmFedAvg rgb glob:", np.median(glob2))
    print("mmFedAvg rgb local:", np.median(spec2))
    #
    # print("mmFedAvg gyro glob:", np.median(glob2))
    # print("mmFedAvg gyro local:", np.median(spec2))

    print("mmFedEKT acce glob:", np.median(glob7))
    print("mmFedEKT acce glob distill epoch 1:", np.median(glob25))
    print("mmFedEKT acce glob distill epoch 3:", np.median(glob26))
    print("mmFedEKT acce glob public ratio 0.1:", np.median(glob37))
    print("mmFedEKT acce glob public ratio 0.5:", np.median(glob38))

    print("mmFedEKT acce glob 1 layer:", np.median(glob21))
    print("mmFedEKT acce local:", np.median(spec7))
    print("mmFedEKT acce local global cls:", np.median(spec22))
    print("mmFedEKT rgb glob:", np.median(glob8))
    print("mmFedEKT rgb glob distill epoch 1:", np.median(glob27))
    print("mmFedEKT rgb glob distill epoch 3:", np.median(glob28))
    print("mmFedEKT rgb glob public ratio 0.1:", np.median(glob39))
    print("mmFedEKT rgb glob public ratio 0.5:", np.median(glob40))
    print("mmFedEKT rgb glob 1 layer:", np.median(glob23))
    print("mmFedEKT rgb local:", np.median(spec8))
    print("mmFedEKT rgb local global cls:", np.median(spec24))

    # print("mmFedEKT gyro glob:", np.median(glob8))
    # print("mmFedEKT gyro local:", np.median(spec8))

    print("---------------ACCE-DEPTH --------------")

    print("mmFedAvg acce glob:", np.median(glob3))
    print("mmFedAvg acce local:", np.median(spec3))

    print("mmFedAvg depth glob:", np.median(glob4))
    print("mmFedAvg depth local:", np.median(spec4))

    print("mmFedEKT acce glob:", np.median(glob9))
    print("mmFedEKT acce glob distill epoch 1:", np.median(glob29))
    print("mmFedEKT acce glob distill epoch 3:", np.median(glob30))
    print("mmFedEKT acce glob public ratio 0.1:", np.median(glob41))
    print("mmFedEKT acce glob public ratio 0.5:", np.median(glob42))
    print("mmFedEKT acce glob 1 layer:", np.median(glob17))
    print("mmFedEKT acce local:", np.median(spec9))
    print("mmFedEKT acce local global cls:", np.median(spec13))

    print("mmFedEKT depth glob:", np.median(glob10))
    print("mmFedEKT depth distill epoch 1:", np.median(glob31))
    print("mmFedEKT depth distill epoch 3:", np.median(glob32))
    print("mmFedEKT depth glob public ratio 0.1:", np.median(glob43))
    print("mmFedEKT depth glob public ratio 0.5:", np.median(glob44))
    print("mmFedEKT depth glob 1 layer:", np.median(glob18))
    print("mmFedEKT depth local:", np.median(spec10))
    print("mmFedEKT depth local global cls:", np.median(spec14))

    print("---------------RGB-DEPTH --------------")
    print("mmFedAvg rgb glob:", np.median(glob5))
    print("mmFedAvg rgb local:", np.median(spec5))

    print("mmFedAvg depth glob:", np.median(glob6))
    print("mmFedAvg depth local:", np.median(spec6))

    print("mmFedEKT rgb glob:", np.median(glob11))
    print("mmFedEKT rgb glob distill epoch 1:", np.median(glob33))
    print("mmFedEKT rgb glob distill epoch 3:", np.median(glob34))
    print("mmFedEKT rgb glob public ratio 0.1:", np.median(glob45))
    print("mmFedEKT rgb glob public ratio 0.5:", np.median(glob46))
    print("mmFedEKT rgb glob 1 layer:", np.median(glob19))
    print("mmFedEKT rgb local:", np.median(spec11))
    print("mmFedEKT rgb local global cls:", np.median(spec15))

    print("mmFedEKT depth glob:", np.median(glob12))
    print("mmFedEKT depth distill epoch 1:", np.median(glob35))
    print("mmFedEKT depth distill epoch 3:", np.median(glob36))
    print("mmFedEKT depth glob public ratio 0.1:", np.median(glob47))
    print("mmFedEKT depth glob public ratio 0.5:", np.median(glob48))
    print("mmFedEKT depth glob 1 layer:", np.median(glob20))
    print("mmFedEKT depth local:", np.median(spec12))
    print("mmFedEKT depth local global cls:", np.median(spec16))


    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                           spec1[np.newaxis, :], spec1[np.newaxis, :], spec1[np.newaxis, :], spec1[np.newaxis, :]), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
                  '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters
def read_files3():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'FedEKD_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha3_eta0.02_beta9_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'FedEKD_ur_fall_LAB_TB_I100_Maacce_Mbrgb_alpha3_eta0.02_beta5_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'FedEKD_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha3_eta0.02_beta5_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'FedEKD_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha3_eta0.02_beta5_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'FedEKD_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha3_eta0.02_beta1_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'FedEKD_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha3_eta0.02_beta3_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]

    print("--------------- SUBSET UR-FALL Project Results --------------")

    print("---------------ACCE-RGB --------------")

    print("FedEKD Acce:", np.median(glob1))
    print("FedEKD Rgb:", np.median(glob2))

    print("---------------ACCE-DEPTH --------------")

    print("FedEKD Acce", np.median(glob3))
    print("FedEKD Depth:", np.median(glob4))

    print("---------------RGB-DEPTH --------------")

    print("FedEKD Rgb:", np.median(glob5))
    print("FedEKD Depth:", np.median(glob6))


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                        ), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters
def read_files4():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'FedEKD_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha1_eta0.02_beta1_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'FedEKD_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha1_eta0.02_beta1_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'FedEKD_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha3_eta0.02_beta1_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'FedEKD_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha3_eta0.02_beta1_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'FedEKD_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha3_eta0.02_beta1_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'FedEKD_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha3_eta0.02_beta1_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5'), 'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]

    print("--------------- SUBSET mHealth Project Results --------------")

    print("---------------ACCE-GYRO --------------")

    print("FedEKD Acce:", np.median(glob1))
    print("FedEKD GYRO:", np.median(glob2))

    print("---------------ACCE-MAGE --------------")

    print("FedEKD Acce", np.median(glob3))
    print("FedEKD MAGE:", np.median(glob4))

    print("---------------GYRO-MAGE --------------")

    print("FedEKD GYRO :", np.median(glob5))
    print("FedEKD MAGE:", np.median(glob6))


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                        ), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters
def read_files_UR_FALL_multimodal_full():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df7 = h5py.File(os.path.join(directory,'mmFedProx_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df8 = h5py.File(os.path.join(directory,'mmFedProx_ur_fall_LAB_TB_I100_Maacce_Mbrgb_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df9 = h5py.File(os.path.join(directory,'mmFedProx_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df10 = h5py.File(os.path.join(directory, 'mmFedProx_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df11 = h5py.File(os.path.join(directory,'mmFedProx_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df12 = h5py.File(os.path.join(directory,'mmFedProx_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df13 = h5py.File(os.path.join(directory,'mmMOON_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df14 = h5py.File(os.path.join(directory, 'mmMOON_ur_fall_LAB_TB_I100_Maacce_Mbrgb_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df15 = h5py.File(os.path.join(directory, 'mmMOON_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df16 = h5py.File(os.path.join(directory, 'mmMOON_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df17 = h5py.File(os.path.join(directory,'mmMOON_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df18 = h5py.File(os.path.join(directory,'mmMOON_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    glob11 = df11['root_test'][Table_index1:Table_index2]
    glob12 = df12['root_test'][Table_index1:Table_index2]
    glob13 = df13['root_test'][Table_index1:Table_index2]
    glob14 = df14['root_test'][Table_index1:Table_index2]
    glob15 = df15['root_test'][Table_index1:Table_index2]
    glob16 = df16['root_test'][Table_index1:Table_index2]
    glob17 = df17['root_test'][Table_index1:Table_index2]
    glob18 = df18['root_test'][Table_index1:Table_index2]

    print("--------------- SUBSET UR Fall Multimodal Results --------------")

    print("---------------ACCE-RGB --------------")

    print("FedMEFKT Acce:", np.median(glob1))
    print("FedMEFKT RGB:", np.median(glob2))
    print("mmFedProx Acce:", np.median(glob7))
    print("mmFedProx RGB:", np.median(glob8))
    print("mmMOON Acce:", np.median(glob13))
    print("mmMOON RGB:", np.median(glob14))

    print("---------------ACCE-DEPTH --------------")

    print("FedMEFKT Acce", np.median(glob3))
    print("FedMEFKT Depth:", np.median(glob4))
    print("mmFedProx Acce", np.median(glob9))
    print("mmFedProx Depth:", np.median(glob10))
    print("mmMOON Acce", np.median(glob15))
    print("mmMOON Depth:", np.median(glob16))

    print("---------------RGB-DEPTH --------------")

    print("FedMEFKT RGB :", np.median(glob5))
    print("FedMEFKT DEPTH:", np.median(glob6))
    print("mmFedProx RGB", np.median(glob11))
    print("mmFedProx DEPTH:", np.median(glob12))
    print("mmMOON RGB", np.median(glob17))
    print("mmMOON DEPTH:", np.median(glob18))


    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                        ), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters


def read_files_UR_FALL_multimodal_unimodal_full():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'mmFedAvg_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'mmFedAvg_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.01_eta0.01_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'mmFedAvg_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'mmFedAvg_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'mmFedAvg_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'mmFedAvg_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df7 = h5py.File(os.path.join(directory,'mmFedProx_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df8 = h5py.File(os.path.join(directory,'mmFedProx_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.05_eta0.05_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df9 = h5py.File(os.path.join(directory,'mmFedProx_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df10 = h5py.File(os.path.join(directory, 'mmFedProx_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df11 = h5py.File(os.path.join(directory,'mmFedProx_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df12 = h5py.File(os.path.join(directory,'mmFedProx_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df13 = h5py.File(os.path.join(directory,'mmMOON_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df14 = h5py.File(os.path.join(directory, 'mmMOON_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.001_eta0.001_beta0_gamma0.0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df15 = h5py.File(os.path.join(directory, 'mmMOON_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df16 = h5py.File(os.path.join(directory, 'mmMOON_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.01_eta0.01_beta0_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df17 = h5py.File(os.path.join(directory,'mmMOON_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df18 = h5py.File(os.path.join(directory,'mmMOON_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.008_eta0.008_beta0.008_gamma0.008_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df19 = h5py.File(os.path.join(directory,'FedMEKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.1_eta0.01_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df20 = h5py.File(os.path.join(directory, 'FedMEKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.02_eta0.05_beta0.0_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df21 = h5py.File(os.path.join(directory,'FedMEKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df22 = h5py.File(os.path.join(directory,'FedMEKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df23 = h5py.File(os.path.join(directory,'FedMEKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df24 = h5py.File(os.path.join(directory,'FedMEKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df25 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df26 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.01_eta0.01_beta0_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df27 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df28 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df29 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df30 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    glob11 = df11['root_test'][Table_index1:Table_index2]
    glob12 = df12['root_test'][Table_index1:Table_index2]
    glob13 = df13['root_test'][Table_index1:Table_index2]
    glob14 = df14['root_test'][Table_index1:Table_index2]
    glob15 = df15['root_test'][Table_index1:Table_index2]
    glob16 = df16['root_test'][Table_index1:Table_index2]
    glob17 = df17['root_test'][Table_index1:Table_index2]
    glob18 = df18['root_test'][Table_index1:Table_index2]
    glob19 = df19['root_test'][Table_index1:Table_index2]
    glob20 = df20['root_test'][Table_index1:Table_index2]
    glob21 = df21['root_test'][Table_index1:Table_index2]
    glob22 = df22['root_test'][Table_index1:Table_index2]
    glob23 = df23['root_test'][Table_index1:Table_index2]
    glob24 = df24['root_test'][Table_index1:Table_index2]
    glob25 = df25['root_test'][Table_index1:Table_index2]
    glob26 = df26['root_test'][Table_index1:Table_index2]
    glob27 = df27['root_test'][Table_index1:Table_index2]
    glob28 = df28['root_test'][Table_index1:Table_index2]
    glob29 = df29['root_test'][Table_index1:Table_index2]
    glob30 = df30['root_test'][Table_index1:Table_index2]

    print("--------------- SUBSET UR Fall Multimodal+Unimodal Results --------------")

    print("---------------ACCE-RGB --------------")

    print("mmFedAvg Acce:", np.median(glob1))
    print("mmFedAvg RGB:", np.median(glob2))
    print("mmFedProx Acce:", np.median(glob7))
    print("mmFedProx RGB:", np.median(glob8))
    print("mmMOON Acce:", np.median(glob13))
    print("mmMOON RGB:", np.median(glob14))
    print("FedMEKT Acce:", np.median(glob19))
    print("FedMEKT RGB:", np.median(glob20))
    print("FedMEFKT Acce:", np.median(glob25))
    print("FedMEFKT RGB:", np.median(glob26))


    print("---------------ACCE-DEPTH --------------")

    print("mmFedAvg Acce", np.median(glob3))
    print("mmFedAvg Depth:", np.median(glob4))
    print("mmFedProx Acce", np.median(glob9))
    print("mmFedProx Depth:", np.median(glob10))
    print("mmMOON Acce", np.median(glob15))
    print("mmMOON Depth:", np.median(glob16))
    print("FedMEKT Acce", np.median(glob21))
    print("FedMEKT Depth:", np.median(glob22))
    print("FedMEFKT Acce", np.median(glob27))
    print("FedMEFKT Depth:", np.median(glob28))

    print("---------------RGB-DEPTH --------------")

    print("mmFedAvg RGB", np.median(glob5))
    print("mmFedAvg DEPTH:", np.median(glob6))
    print("mmFedProx RGB", np.median(glob11))
    print("mmFedProx DEPTH:", np.median(glob12))
    print("mmMOON RGB", np.median(glob17))
    print("mmMOON DEPTH:", np.median(glob18))
    print("FedMEKT RGB :", np.median(glob23))
    print("FedMEKT DEPTH:", np.median(glob24))
    print("FedMEFKT RGB :", np.median(glob29))
    print("FedMEFKT DEPTH:", np.median(glob30))


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                        ), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters

def read_files_UR_FALL_proxy_data():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'), 'r')
    df7 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'),'r')
    df8 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'),'r')
    df9 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'),'r')
    df10 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'),'r')
    df11 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'),'r')
    df12 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'),'r')
    df13 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'), 'r')
    df14 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'), 'r')
    df15 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.01_eta0.01_beta0_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'), 'r')
    df16 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.01_eta0.01_beta0_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'), 'r')
    df17 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'),'r')
    df18 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'), 'r')
    df19 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'),'r')
    df20 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'),'r')
    df21 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'),'r')
    df22 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'), 'r')
    df23 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.5.h5'),'r')
    df24 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse_publicratio0.1.h5'), 'r')

    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    glob11 = df11['root_test'][Table_index1:Table_index2]
    glob12 = df12['root_test'][Table_index1:Table_index2]
    glob13 = df13['root_test'][Table_index1:Table_index2]
    glob14 = df14['root_test'][Table_index1:Table_index2]
    glob15 = df15['root_test'][Table_index1:Table_index2]
    glob16 = df16['root_test'][Table_index1:Table_index2]
    glob17 = df17['root_test'][Table_index1:Table_index2]
    glob18 = df18['root_test'][Table_index1:Table_index2]
    glob19 = df19['root_test'][Table_index1:Table_index2]
    glob20 = df20['root_test'][Table_index1:Table_index2]
    glob21 = df21['root_test'][Table_index1:Table_index2]
    glob22 = df22['root_test'][Table_index1:Table_index2]
    glob23 = df23['root_test'][Table_index1:Table_index2]
    glob24 = df24['root_test'][Table_index1:Table_index2]


    print("--------------- SUBSET UR Fall Proxy Data Results (Multimodal) --------------")

    print("---------------ACCE-RGB --------------")


    print("FedMEFKT Acce 50%:", np.median(glob1))
    print("FedMEFKT RGB 50%:", np.median(glob3))
    print("FedMEFKT Acce 10%:", np.median(glob2))
    print("FedMEFKT RGB 10%:", np.median(glob4))


    print("---------------ACCE-DEPTH --------------")

    print("FedMEFKT Acce 50%", np.median(glob5))
    print("FedMEFKT Depth 50%:", np.median(glob7))
    print("FedMEFKT Acce 10%", np.median(glob6))
    print("FedMEFKT Depth 10%:", np.median(glob8))

    print("---------------RGB-DEPTH --------------")

    print("FedMEFKT RGB 50% :", np.median(glob9))
    print("FedMEFKT DEPTH 50%:", np.median(glob11))
    print("FedMEFKT RGB 10% :", np.median(glob10))
    print("FedMEFKT DEPTH 10%:", np.median(glob12))

    print("--------------- SUBSET UR Fall Proxy Data Results (Multimodal+Unimodal) --------------")

    print("---------------ACCE-RGB --------------")

    print("FedMEFKT Acce 50%:", np.median(glob13))
    print("FedMEFKT RGB 50%:", np.median(glob15))
    print("FedMEFKT Acce 10%:", np.median(glob14))
    print("FedMEFKT RGB 10%:", np.median(glob16))

    print("---------------ACCE-DEPTH --------------")

    print("FedMEFKT Acce 50%", np.median(glob17))
    print("FedMEFKT Depth 50%:", np.median(glob19))
    print("FedMEFKT Acce 10%", np.median(glob18))
    print("FedMEFKT Depth 10%:", np.median(glob20))

    print("---------------RGB-DEPTH --------------")

    print("FedMEFKT RGB 50% :", np.median(glob21))
    print("FedMEFKT DEPTH 50%:", np.median(glob23))
    print("FedMEFKT RGB 10% :", np.median(glob22))
    print("FedMEFKT DEPTH 10%:", np.median(glob24))


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                        ), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters


def read_files_UR_FALL_R_steps():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df7 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df8 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df9 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df10 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df11 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df12 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df13 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df14 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df15 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df16 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df17 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df18 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df19 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df20 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df21 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df22 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df23 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df24 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')

    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    glob11 = df11['root_test'][Table_index1:Table_index2]
    glob12 = df12['root_test'][Table_index1:Table_index2]
    glob13 = df13['root_test'][Table_index1:Table_index2]
    glob14 = df14['root_test'][Table_index1:Table_index2]
    glob15 = df15['root_test'][Table_index1:Table_index2]
    glob16 = df16['root_test'][Table_index1:Table_index2]
    glob17 = df17['root_test'][Table_index1:Table_index2]
    glob18 = df18['root_test'][Table_index1:Table_index2]
    glob19 = df19['root_test'][Table_index1:Table_index2]
    glob20 = df20['root_test'][Table_index1:Table_index2]
    glob21 = df21['root_test'][Table_index1:Table_index2]
    glob22 = df22['root_test'][Table_index1:Table_index2]
    glob23 = df23['root_test'][Table_index1:Table_index2]
    glob24 = df24['root_test'][Table_index1:Table_index2]


    print("--------------- SUBSET UR Fall R Step Results (Multimodal) --------------")

    print("---------------ACCE-RGB --------------")


    print("FedMEFKT Acce R=1:", np.median(glob1))
    print("FedMEFKT RGB R=1:", np.median(glob3))
    print("FedMEFKT Acce R=3:", np.median(glob2))
    print("FedMEFKT RGB R=3:", np.median(glob4))


    print("---------------ACCE-DEPTH --------------")

    print("FedMEFKT Acce R=1", np.median(glob5))
    print("FedMEFKT Depth R=1:", np.median(glob7))
    print("FedMEFKT Acce R=3", np.median(glob6))
    print("FedMEFKT Depth R=3:", np.median(glob8))

    print("---------------RGB-DEPTH --------------")

    print("FedMEFKT RGB R=1 :", np.median(glob9))
    print("FedMEFKT DEPTH R=1:", np.median(glob11))
    print("FedMEFKT RGB R=3 :", np.median(glob10))
    print("FedMEFKT DEPTH R=3:", np.median(glob12))

    print("--------------- SUBSET UR Fall R Step Results (Multimodal+Unimodal) --------------")

    print("---------------ACCE-RGB --------------")

    print("FedMEFKT Acce R=1:", np.median(glob13))
    print("FedMEFKT RGB R=1:", np.median(glob15))
    print("FedMEFKT Acce R=3:", np.median(glob14))
    print("FedMEFKT RGB R=3:", np.median(glob16))

    print("---------------ACCE-DEPTH --------------")

    print("FedMEFKT Acce R=1", np.median(glob17))
    print("FedMEFKT Depth R=1:", np.median(glob19))
    print("FedMEFKT Acce R=3", np.median(glob18))
    print("FedMEFKT Depth R=3:", np.median(glob20))

    print("---------------RGB-DEPTH --------------")

    print("FedMEFKT RGB R=1 :", np.median(glob21))
    print("FedMEFKT DEPTH R=1:", np.median(glob23))
    print("FedMEFKT RGB R=3 :", np.median(glob22))
    print("FedMEFKT DEPTH R=3:", np.median(glob24))


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                        ), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters

def read_files_UR_FALL_local_epoch():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df7 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df8 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df9 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df10 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df11 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df12 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df13 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df14 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df15 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.01_eta0.01_beta0_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df16 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.01_eta0.01_beta0_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df17 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df18 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.1_eta0.1_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df19 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df20 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.04_eta0.04_beta0.01_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df21 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df22 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.04_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df23 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch1_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df24 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.02_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch3_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')

    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    glob11 = df11['root_test'][Table_index1:Table_index2]
    glob12 = df12['root_test'][Table_index1:Table_index2]
    glob13 = df13['root_test'][Table_index1:Table_index2]
    glob14 = df14['root_test'][Table_index1:Table_index2]
    glob15 = df15['root_test'][Table_index1:Table_index2]
    glob16 = df16['root_test'][Table_index1:Table_index2]
    glob17 = df17['root_test'][Table_index1:Table_index2]
    glob18 = df18['root_test'][Table_index1:Table_index2]
    glob19 = df19['root_test'][Table_index1:Table_index2]
    glob20 = df20['root_test'][Table_index1:Table_index2]
    glob21 = df21['root_test'][Table_index1:Table_index2]
    glob22 = df22['root_test'][Table_index1:Table_index2]
    glob23 = df23['root_test'][Table_index1:Table_index2]
    glob24 = df24['root_test'][Table_index1:Table_index2]


    print("--------------- SUBSET UR Fall Local Epoch (Multimodal) --------------")

    print("---------------ACCE-RGB --------------")


    print("FedMEFKT Acce N=1:", np.median(glob1))
    print("FedMEFKT RGB N=1:", np.median(glob3))
    print("FedMEFKT Acce N=3:", np.median(glob2))
    print("FedMEFKT RGB N=3:", np.median(glob4))


    print("---------------ACCE-DEPTH --------------")

    print("FedMEFKT Acce N=1", np.median(glob5))
    print("FedMEFKT Depth N=1:", np.median(glob7))
    print("FedMEFKT Acce N=3", np.median(glob6))
    print("FedMEFKT Depth N=3:", np.median(glob8))

    print("---------------RGB-DEPTH --------------")

    print("FedMEFKT RGB N=1 :", np.median(glob9))
    print("FedMEFKT DEPTH N=1:", np.median(glob11))
    print("FedMEFKT RGB N=3 :", np.median(glob10))
    print("FedMEFKT DEPTH N=3:", np.median(glob12))

    print("--------------- SUBSET UR Fall Local epoch Results (Multimodal+Unimodal) --------------")

    print("---------------ACCE-RGB --------------")

    print("FedMEFKT Acce N=1:", np.median(glob13))
    print("FedMEFKT RGB N=1:", np.median(glob15))
    print("FedMEFKT Acce N=3:", np.median(glob14))
    print("FedMEFKT RGB N=3:", np.median(glob16))

    print("---------------ACCE-DEPTH --------------")

    print("FedMEFKT Acce N=1", np.median(glob17))
    print("FedMEFKT Depth N=1:", np.median(glob19))
    print("FedMEFKT Acce N=3", np.median(glob18))
    print("FedMEFKT Depth N=3:", np.median(glob20))

    print("---------------RGB-DEPTH --------------")

    print("FedMEFKT RGB N=1 :", np.median(glob21))
    print("FedMEFKT DEPTH N=1:", np.median(glob23))
    print("FedMEFKT RGB N=3 :", np.median(glob22))
    print("FedMEFKT DEPTH N=3:", np.median(glob24))


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                        ), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters

def read_files_UR_FALL_ablation():
    directory = "./results_fig/"
    df = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0_eta0.1_beta0_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df2 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.1_eta0_beta0.02_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df3 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0_eta0.04_beta0_gamma0.01_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df4 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbrgb_alpha0.04_eta0_beta0.01_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df5 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0_eta0.1_beta0_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df6 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0_beta0.04_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df7 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0_eta0.1_beta0_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df8 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Maacce_Mbdepth_alpha0.1_eta0_beta0.04_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df9 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Margb_Mbdepth_alpha0_eta0.04_beta0_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df10 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0_beta0.04_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df11 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A0_B0_AB30_I100_Margb_Mbdepth_alpha0_eta0.04_beta0_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df12 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A0_B0_AB30_I100_Margb_Mbdepth_alpha0.1_eta0_beta0.02_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df13 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0_eta0.1_beta0_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df14 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.1_eta0_beta0.02_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df15 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0_eta0.01_beta0_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df16 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbrgb_alpha0.01_eta0_beta0_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df17 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0_eta0.1_beta0_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df18 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.1_eta0_beta0.04_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df19 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0_eta0.1_beta0_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df20 = h5py.File(os.path.join(directory, 'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Maacce_Mbdepth_alpha0.1_eta0_beta0.04_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df21 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0_eta0.04_beta0_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df22 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TA_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0_beta0.04_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')
    df23 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0_eta0.1_beta0_gamma0.02_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'),'r')
    df24 = h5py.File(os.path.join(directory,'FedMEFKT_ur_fall_LAB_TB_A10_B10_AB10_I100_Margb_Mbdepth_alpha0.1_eta0_beta0.02_gamma0_SSTrue_gmKL_lmKL_ratio0.11_depoch2_lepoch2_onelayerFalse_globalclsDrFalse_publicratio1.h5'), 'r')

    # stop1 = df['stop1'][:]

    # glob1 = df['root_test'][Start_index:]
    # gen1 = df['cg_avg_data_test'][Start_index:]
    # glob2 = df2['root_test'][Start_index:]
    # gen2 = df2['cg_avg_data_test'][Start_index:]
    # glob3 = df3['root_test'][Start_index:]
    # gen3 = df3['cg_avg_data_test'][Start_index:]
    # glob4 = df4['root_test'][Start_index:]
    # gen4 = df4['cg_avg_data_test'][Start_index:]


    glob1 = df['root_test'][Table_index1:Table_index2]
    glob2 = df2['root_test'][Table_index1:Table_index2]
    glob3 = df3['root_test'][Table_index1:Table_index2]
    glob4 = df4['root_test'][Table_index1:Table_index2]
    glob5 = df5['root_test'][Table_index1:Table_index2]
    glob6 = df6['root_test'][Table_index1:Table_index2]
    glob7 = df7['root_test'][Table_index1:Table_index2]
    glob8 = df8['root_test'][Table_index1:Table_index2]
    glob9 = df9['root_test'][Table_index1:Table_index2]
    glob10 = df10['root_test'][Table_index1:Table_index2]
    glob11 = df11['root_test'][Table_index1:Table_index2]
    glob12 = df12['root_test'][Table_index1:Table_index2]
    glob13 = df13['root_test'][Table_index1:Table_index2]
    glob14 = df14['root_test'][Table_index1:Table_index2]
    glob15 = df15['root_test'][Table_index1:Table_index2]
    glob16 = df16['root_test'][Table_index1:Table_index2]
    glob17 = df17['root_test'][Table_index1:Table_index2]
    glob18 = df18['root_test'][Table_index1:Table_index2]
    glob19 = df19['root_test'][Table_index1:Table_index2]
    glob20 = df20['root_test'][Table_index1:Table_index2]
    glob21 = df21['root_test'][Table_index1:Table_index2]
    glob22 = df22['root_test'][Table_index1:Table_index2]
    glob23 = df23['root_test'][Table_index1:Table_index2]
    glob24 = df24['root_test'][Table_index1:Table_index2]


    print("--------------- SUBSET UR Fall Ablation (Multimodal) --------------")

    print("---------------ACCE-RGB --------------")


    print("FedMEFKT Acce w/o local EKT:", np.median(glob1))
    print("FedMEFKT RGB w/o local EKT:", np.median(glob3))
    print("FedMEFKT Acce w/o global EKT:", np.median(glob2))
    print("FedMEFKT RGB w/o global EKT:", np.median(glob4))


    print("---------------ACCE-DEPTH --------------")

    print("FedMEFKT Acce w/o local EKT", np.median(glob5))
    print("FedMEFKT Depth w/o local EKT:", np.median(glob7))
    print("FedMEFKT Acce w/o global EKT", np.median(glob6))
    print("FedMEFKT Depth w/o global EKT:", np.median(glob8))

    print("---------------RGB-DEPTH --------------")

    print("FedMEFKT RGB w/o local EKT :", np.median(glob9))
    print("FedMEFKT DEPTH w/o local EKT:", np.median(glob11))
    print("FedMEFKT RGB w/o global EKT :", np.median(glob10))
    print("FedMEFKT DEPTH w/o global EKT:", np.median(glob12))

    print("--------------- SUBSET UR Fall Ablation (Multimodal+Unimodal) --------------")

    print("---------------ACCE-RGB --------------")

    print("FedMEFKT Acce w/o local EKT:", np.median(glob13))
    print("FedMEFKT RGB w/o local EKT:", np.median(glob15))
    print("FedMEFKT Acce w/o global EKT:", np.median(glob14))
    print("FedMEFKT RGB w/o global EKT:", np.median(glob16))

    print("---------------ACCE-DEPTH --------------")

    print("FedMEFKT Acce w/o local EKT", np.median(glob17))
    print("FedMEFKT Depth w/o local EKT:", np.median(glob19))
    print("FedMEFKT Acce w/o global EKT", np.median(glob18))
    print("FedMEFKT Depth w/o global EKT:", np.median(glob20))

    print("---------------RGB-DEPTH --------------")

    print("FedMEFKT RGB w/o local EKT :", np.median(glob21))
    print("FedMEFKT DEPTH w/o local EKT:", np.median(glob23))
    print("FedMEFKT RGB w/o global EKT :", np.median(glob22))
    print("FedMEFKT DEPTH w/o global EKT:", np.median(glob24))


    # print(glob1)
    # print(gen1)
    # stop4 = df['stop4'][:]
    # rs_Objs = df['rs_Objs'][:]
    # print("Avg BCD:",np.average(stop1))
    # print("glob perf:", np.median(glob1))
    # print("gen perf:", np.median(gen1))
    # print("Avg JP-miADMM ES:", np.median(stop4))
    # print("Obj1:",rs_Objs[:,1])
    # print("Obj2:", rs_Objs[:, 2])
    # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
    #
    # data = np.concatenate((gen,glob), axis=1)
    data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
                        ), axis=0)
    # print(data.transpose())
    iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal']
    # data = np.concatenate((stop1, stop3), axis=1)
    # iters_cols =['Centralized','Decentralized']
    df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
    df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
    # print(df_iters)
    return df_iters
# def read_files4():
#     directory = "./results_fig/"
#     df = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fTrue_a0.02_b0.25_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar10_I100_sTrue_fFalse_a0.01_b0.1_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0_b0_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
#     # stop1 = df['stop1'][:]
#
#     # glob1 = df['root_test'][Start_index:]
#     # gen1 = df['cg_avg_data_test'][Start_index:]
#     # glob2 = df2['root_test'][Start_index:]
#     # gen2 = df2['cg_avg_data_test'][Start_index:]
#     # glob3 = df3['root_test'][Start_index:]
#     # gen3 = df3['cg_avg_data_test'][Start_index:]
#     # glob4 = df4['root_test'][Start_index:]
#     # gen4 = df4['cg_avg_data_test'][Start_index:]
#
#
#     glob1 = df['root_test'][Table_index1:Table_index2]
#     gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
#     spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
#     glob2 = df2['root_test'][Table_index1:Table_index2]
#     gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
#     spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
#     glob3 = df3['root_test'][Table_index1:Table_index2]
#     gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
#     spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
#     glob4 = df4['root_test'][Table_index1:Table_index2]
#     gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
#     spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
#     glob5 = df5['root_test'][Table_index1:Table_index2]
#     gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
#     spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]
#
#     print("--------------- FIXED CIFAR-10 RESULTS --------------")
#
#     print("CDKT Rep KL-N glob:", np.median(glob1))
#     print("CDKT Rep KL-N gen:", np.median(gen1))
#     print("CDKT Rep KL-N spec:", np.median(spec1))
#     print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)
#
#     print("CDKT Rep Full KL-N glob:", np.median(glob2))
#     print("CDKT Rep Full KL-N gen:", np.median(gen2))
#     print("CDKT Rep Full KL-N spec:", np.median(spec2))
#     print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)
#
#     print("CDKT Full KL-N glob:", np.median(glob3))
#     print("CDKT Full KL-N gen:", np.median(gen3))
#     print("CDKT Full KL-N spec:", np.median(spec3))
#     print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)
#
#     print("fedavg glob:", np.median(glob4))
#     print("fedavg gen:", np.median(gen4))
#     print("fedavg spec:", np.median(spec4))
#     print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)
#
#     print("CDKT no transfer glob:", np.median(glob5))
#     print("CDKT no transfer gen:", np.median(gen5))
#     print("CDKT no transfer spec:", np.median(spec5))
#     print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)
#
#
#     # print(glob1)
#     # print(gen1)
#     # stop4 = df['stop4'][:]
#     # rs_Objs = df['rs_Objs'][:]
#     # print("Avg BCD:",np.average(stop1))
#     # print("glob perf:", np.median(glob1))
#     # print("gen perf:", np.median(gen1))
#     # print("Avg JP-miADMM ES:", np.median(stop4))
#     # print("Obj1:",rs_Objs[:,1])
#     # print("Obj2:", rs_Objs[:, 2])
#     # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
#     #
#     # data = np.concatenate((gen,glob), axis=1)
#     data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
#                            gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
#     # print(data.transpose())
#     iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
#                   '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
#     # data = np.concatenate((stop1, stop3), axis=1)
#     # iters_cols =['Centralized','Decentralized']
#     df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
#     df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
#     # print(df_iters)
#     return df_iters
#
#
# def read_files5():
#     directory = "./results_fig/"
#     df = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.5_b0.14_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
#     # df = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'),'r')
#     df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0.75_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
#
#     df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fTrue_a0.6_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar10_I100_sTrue_fFalse_a0.02_b0.25_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar10_I100_sTrue_fFalse_a0_b0_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
#     # stop1 = df['stop1'][:]
#     #
#     # glob1 = df['root_test'][Start_index:]
#     # gen1 = df['cg_avg_data_test'][Start_index:]
#     # glob2 = df2['root_test'][Start_index:]
#     # gen2 = df2['cg_avg_data_test'][Start_index:]
#     # glob3 = df3['root_test'][Start_index:]
#     # gen3 = df3['cg_avg_data_test'][Start_index:]
#     # glob4 = df4['root_test'][Start_index:]
#     # gen4 = df4['cg_avg_data_test'][Start_index:]
#
#     glob1 = df['root_test'][Table_index1:Table_index2]
#     gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
#     spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
#     glob2 = df2['root_test'][Table_index1:Table_index2]
#     gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
#     spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
#     glob3 = df3['root_test'][Table_index1:Table_index2]
#     gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
#     spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
#     glob4 = df4['root_test'][Table_index1:Table_index2]
#     gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
#     spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
#     glob5 = df5['root_test'][Table_index1:Table_index2]
#     gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
#     spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]
#
#     print("--------------- SUBSET CIFAR-10 RESULTS --------------")
#
#     print("CDKT Rep KL-N glob:", np.median(glob1))
#     print("CDKT Rep KL-N gen:", np.median(gen1))
#     print("CDKT Rep KL-N spec:", np.median(spec1))
#     print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)
#
#     print("CDKT Rep Full KL-N glob:", np.median(glob2))
#     print("CDKT Rep Full KL-N gen:", np.median(gen2))
#     print("CDKT Rep Full KL-N spec:", np.median(spec2))
#     print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)
#
#     print("CDKT Full KL-N glob:", np.median(glob3))
#     print("CDKT Full KL-N gen:", np.median(gen3))
#     print("CDKT Full KL-N spec:", np.median(spec3))
#     print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)
#
#     print("fedavg glob:", np.median(glob4))
#     print("fedavg gen:", np.median(gen4))
#     print("fedavg spec:", np.median(spec4))
#     print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)
#
#     print("CDKT no transfer glob:", np.median(glob5))
#     print("CDKT no transfer gen:", np.median(gen5))
#     print("CDKT no transfer spec:", np.median(spec5))
#     print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)
#
#
#     # print(glob1)
#     # print(gen1)
#     # stop4 = df['stop4'][:]
#     # rs_Objs = df['rs_Objs'][:]
#     # print("Avg BCD:",np.average(stop1))
#     # print("glob perf:", np.median(glob1))
#     # print("gen perf:", np.median(gen1))
#     # print("Avg JP-miADMM ES:", np.median(stop4))
#     # print("Obj1:",rs_Objs[:,1])
#     # print("Obj2:", rs_Objs[:, 2])
#     # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
#     #
#     # data = np.concatenate((gen,glob), axis=1)
#     data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
#                            gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
#     # print(data.transpose())
#     iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
#                   '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
#     # data = np.concatenate((stop1, stop3), axis=1)
#     # iters_cols =['Centralized','Decentralized']
#     df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
#     df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
#     # print(df_iters)
#     return df_iters
# def read_files6():
#     directory = "./results_fig/"
#     df = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0.1_b0.04_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.5_b0.2_RFFalse_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSFalse_accFalse_gmKL_lmNorm2.h5'), 'r')
#     # stop1 = df['stop1'][:]
#
#     # glob1 = df['root_test'][Start_index:]
#     # gen1 = df['cg_avg_data_test'][Start_index:]
#     # glob2 = df2['root_test'][Start_index:]
#     # gen2 = df2['cg_avg_data_test'][Start_index:]
#     # glob3 = df3['root_test'][Start_index:]
#     # gen3 = df3['cg_avg_data_test'][Start_index:]
#     # glob4 = df4['root_test'][Start_index:]
#     # gen4 = df4['cg_avg_data_test'][Start_index:]
#
#
#     glob1 = df['root_test'][Table_index1:Table_index2]
#     gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
#     spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
#     glob2 = df2['root_test'][Table_index1:Table_index2]
#     gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
#     spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
#     glob3 = df3['root_test'][Table_index1:Table_index2]
#     gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
#     spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
#     glob4 = df4['root_test'][Table_index1:Table_index2]
#     gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
#     spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
#     glob5 = df5['root_test'][Table_index1:Table_index2]
#     gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
#     spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]
#
#     print("--------------- FIXED CIFAR-100 RESULTS --------------")
#
#     print("CDKT Rep KL-N glob:", np.median(glob1))
#     print("CDKT Rep KL-N gen:", np.median(gen1))
#     print("CDKT Rep KL-N spec:", np.median(spec1))
#     print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)
#
#     print("CDKT Rep Full KL-N glob:", np.median(glob2))
#     print("CDKT Rep Full KL-N gen:", np.median(gen2))
#     print("CDKT Rep Full KL-N spec:", np.median(spec2))
#     print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)
#
#     print("CDKT Full KL-N glob:", np.median(glob3))
#     print("CDKT Full KL-N gen:", np.median(gen3))
#     print("CDKT Full KL-N spec:", np.median(spec3))
#     print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)
#
#     print("fedavg glob:", np.median(glob4))
#     print("fedavg gen:", np.median(gen4))
#     print("fedavg spec:", np.median(spec4))
#     print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)
#
#     print("CDKT no transfer glob:", np.median(glob5))
#     print("CDKT no transfer gen:", np.median(gen5))
#     print("CDKT no transfer spec:", np.median(spec5))
#     print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)
#
#
#     # print(glob1)
#     # print(gen1)
#     # stop4 = df['stop4'][:]
#     # rs_Objs = df['rs_Objs'][:]
#     # print("Avg BCD:",np.average(stop1))
#     # print("glob perf:", np.median(glob1))
#     # print("gen perf:", np.median(gen1))
#     # print("Avg JP-miADMM ES:", np.median(stop4))
#     # print("Obj1:",rs_Objs[:,1])
#     # print("Obj2:", rs_Objs[:, 2])
#     # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
#     #
#     # data = np.concatenate((gen,glob), axis=1)
#
#
#     data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
#                            gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
#     # print(data.transpose())
#     iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
#                   '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
#     # data = np.concatenate((stop1, stop3), axis=1)
#     # iters_cols =['Centralized','Decentralized']
#     df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
#     df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
#     # print(df_iters)
#     return df_iters
#
#
# def read_files7():
#     directory = "./results_fig/"
#     df = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.45_b0.09_RFFalse_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
#     # df = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'),'r')
#     df2 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
#
#     df3 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0.8_b0.1_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df4 = h5py.File(os.path.join(directory, 'fedavg_Cifar100_I100_sTrue_fFalse_a0.8_b0.07_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
#     df5 = h5py.File(os.path.join(directory, 'CDKT_Cifar100_I100_sTrue_fTrue_a0_b0_RFTrue_SSTrue_accFalse_gmKL_lmNorm2.h5'), 'r')
#     # stop1 = df['stop1'][:]
#     #
#     # glob1 = df['root_test'][Start_index:]
#     # gen1 = df['cg_avg_data_test'][Start_index:]
#     # glob2 = df2['root_test'][Start_index:]
#     # gen2 = df2['cg_avg_data_test'][Start_index:]
#     # glob3 = df3['root_test'][Start_index:]
#     # gen3 = df3['cg_avg_data_test'][Start_index:]
#     # glob4 = df4['root_test'][Start_index:]
#     # gen4 = df4['cg_avg_data_test'][Start_index:]
#
#     glob1 = df['root_test'][Table_index1:Table_index2]
#     gen1 = df['cg_avg_data_test'][Table_index1:Table_index2]
#     spec1 = df['cs_avg_data_test'][Table_index1:Table_index2]
#     glob2 = df2['root_test'][Table_index1:Table_index2]
#     gen2 = df2['cg_avg_data_test'][Table_index1:Table_index2]
#     spec2 = df2['cs_avg_data_test'][Table_index1:Table_index2]
#     glob3 = df3['root_test'][Table_index1:Table_index2]
#     gen3 = df3['cg_avg_data_test'][Table_index1:Table_index2]
#     spec3 = df3['cs_avg_data_test'][Table_index1:Table_index2]
#     glob4 = df4['root_test'][Table_index1:Table_index2]
#     gen4 = df4['cg_avg_data_test'][Table_index1:Table_index2]
#     spec4 = df4['cs_avg_data_test'][Table_index1:Table_index2]
#     glob5 = df5['root_test'][Table_index1:Table_index2]
#     gen5 = df5['cg_avg_data_test'][Table_index1:Table_index2]
#     spec5 = df5['cs_avg_data_test'][Table_index1:Table_index2]
#
#     print("--------------- SUBSET CIFAR-100 RESULTS --------------")
#
#     print("CDKT Rep KL-N glob:", np.median(glob1))
#     print("CDKT Rep KL-N gen:", np.median(gen1))
#     print("CDKT Rep KL-N spec:", np.median(spec1))
#     print("CDKT Rep KL-N personalized:", (np.median(spec1) + np.median(gen1)) / 2)
#
#     print("CDKT Rep Full KL-N glob:", np.median(glob2))
#     print("CDKT Rep Full KL-N gen:", np.median(gen2))
#     print("CDKT Rep Full KL-N spec:", np.median(spec2))
#     print("CDKT Rep Full KL-N personalized:", (np.median(spec2) + np.median(gen2)) / 2)
#
#     print("CDKT Full KL-N glob:", np.median(glob3))
#     print("CDKT Full KL-N gen:", np.median(gen3))
#     print("CDKT Full KL-N spec:", np.median(spec3))
#     print("CDKT Full KL-N personalized:", (np.median(spec3) + np.median(gen3)) / 2)
#
#     print("fedavg glob:", np.median(glob4))
#     print("fedavg gen:", np.median(gen4))
#     print("fedavg spec:", np.median(spec4))
#     print("fedavg personalized:", (np.median(spec4) + np.median(gen4)) / 2)
#
#     print("CDKT no transfer glob:", np.median(glob5))
#     print("CDKT no transfer gen:", np.median(gen5))
#     print("CDKT no transfer spec:", np.median(spec5))
#     print("CDKT no transfer personalized:", (np.median(spec5) + np.median(gen5)) / 2)
#
#
#     # print(glob1)
#     # print(gen1)
#     # stop4 = df['stop4'][:]
#     # rs_Objs = df['rs_Objs'][:]
#     # print("Avg BCD:",np.average(stop1))
#     # print("glob perf:", np.median(glob1))
#     # print("gen perf:", np.median(gen1))
#     # print("Avg JP-miADMM ES:", np.median(stop4))
#     # print("Obj1:",rs_Objs[:,1])
#     # print("Obj2:", rs_Objs[:, 2])
#     # print("Obj_ratio:", np.average(rs_Objs[:, 2] / rs_Objs[:, 0]) * 100)
#     #
#     # data = np.concatenate((gen,glob), axis=1)
#
#
#     data = np.concatenate((glob1[np.newaxis, :], glob2[np.newaxis, :], glob3[np.newaxis, :], glob4[np.newaxis, :],
#                            gen1[np.newaxis, :], gen2[np.newaxis, :], gen3[np.newaxis, :], gen4[np.newaxis, :]), axis=0)
#     # print(data.transpose())
#     iters_cols = ['(Rep+KL-N)\nGlobal','(RepFull+KL-N)\nGlobal','(Full+KL-N)\nGlobal','FedAvg\nGlobal',
#                   '(Rep+KL-N)\nC-Gen','(RepFull+KL-N)\nC-Gen','(Full+KL-N)\nC-Gen','FedAvg\nC-Gen']
#     # data = np.concatenate((stop1, stop3), axis=1)
#     # iters_cols =['Centralized','Decentralized']
#     df_iters = pd.DataFrame(data.transpose()*100, columns=iters_cols)
#     df_iters = pd.melt(df_iters, var_name='Algorithm', value_name='Accuracy')
#     # print(df_iters)
#     return df_iters
def communication_cost_plot():
    a =0
plot_final()