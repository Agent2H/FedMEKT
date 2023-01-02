import h5py as hf
import numpy as np
from Setting import *
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

plt.rcParams.update({'font.size': 16})  #font size 10 12 14 16 main 16
plt.rcParams['lines.linewidth'] = 3
YLim=0
#Global variable
markers_on = 10 #maker only at x = markers_on[], default 10
OUT_TYPE = ".pdf" #.eps, .pdf, .jpeg #output figure type

color = {
    "gen": "royalblue",
    "cspe": "forestgreen",
    "cgen": "red",
    "c": "cyan",
    "gspe": "darkorange",  #magenta
    "gg": "yellow",
    "ggen": "darkviolet",
    "w": "white",
    "g":"green"
}
marker = {
    "gen": "8",
    "gspe": "s",
    "ggen": "P",
    "cspe": "p",
    "cgen": "*"
}

def read_data(file_name = "../results/untitled.h5"):
    dic_data = {}
    with hf.File(file_name, "r") as f:
        # List all groups
        #print("Keys: %s" % f.keys())
        for key in f.keys():
            try:
                dic_data[key] = f[key][:]
            except:
                dic_data[key] = f[key]
    return  dic_data



def plot_ur_fall():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 4.4))

    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    mmfedavgAcceRgb_testAcce = read_data(RS_PATH + name['mmfedavg_acce_rgb_testacce'])
    mmfedavgAcceRgb_testRgb = read_data(RS_PATH + name['mmfedavg_acce_rgb_testrgb'])
    mmfedavgAcceDepth_testDepth = read_data(RS_PATH + name['mmfedavg_acce_depth_testdepth'])
    mmfedavgAcceDepth_testAcce = read_data(RS_PATH + name['mmfedavg_acce_depth_testacce'])
    mmfedavgRgbDepth_testRgb = read_data(RS_PATH + name['mmfedavg_rgb_depth_testrgb'])
    mmfedavgRgbDepth_testDepth = read_data(RS_PATH + name['mmfedavg_rgb_depth_testdepth'])
    mmfedEKTAcceRgb_testAcce = read_data(RS_PATH + name['mmfedEKT_acce_rgb_testacce'])
    mmfedEKTAcceRgb_testRgb = read_data(RS_PATH + name['mmfedEKT_acce_rgb_testrgb'])
    mmfedEKTAcceDepth_testDepth = read_data(RS_PATH + name['mmfedEKT_acce_depth_testdepth'])
    mmfedEKTAcceDepth_testAcce = read_data(RS_PATH + name['mmfedEKT_acce_depth_testacce'])
    mmfedEKTRgbDepth_testRgb = read_data(RS_PATH + name['mmfedEKT_rgb_depth_testrgb'])
    mmfedEKTRgbDepth_testDepth = read_data(RS_PATH + name['mmfedEKT_rgb_depth_testdepth'])


    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(mmfedavgAcceRgb_testAcce['root_test'], label="Mm-FedAvg(Acce)",  color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(mmfedavgAcceRgb_testRgb['root_test'])), mmfedavgAcceRgb_testRgb['root_test'],
             color=color["c"], linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="Mm-FedAvg(Rgb)")

    ax1.plot(np.arange(len(mmfedEKTAcceRgb_testAcce['root_test'])), mmfedEKTAcceRgb_testAcce['root_test'], color=color["g"],
             marker=marker["cgen"], markevery=markers_on,
             label="FedMEKT(Acce)")
    ax1.plot(np.arange(len(mmfedEKTAcceRgb_testRgb['root_test'])), mmfedEKTAcceRgb_testRgb['root_test'],
             color=color["cgen"], linestyle='dashed',
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Rgb)")



    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 0.8)
    ax1.set_title(DATASET+" "+ "Acce-Rgb")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2


    # print("CDKTKL Global", f_data['root_test'][12])
    # print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(np.arange(len( mmfedavgAcceDepth_testDepth['root_test'])),  mmfedavgAcceDepth_testDepth['root_test'],
             color=color["gen"],
             marker=marker["gen"], markevery=markers_on,
             label="Mm-FedAvg(Depth)")
    ax2.plot(np.arange(len( mmfedavgAcceDepth_testAcce['root_test'])),  mmfedavgAcceDepth_testAcce['root_test'],
             color=color["c"], linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="Mm-FedAvg(Acce)")
    ax2.plot(np.arange(len( mmfedEKTAcceDepth_testDepth['root_test'])),  mmfedEKTAcceDepth_testDepth['root_test'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Depth)")
    ax2.plot(np.arange(len( mmfedEKTAcceDepth_testAcce['root_test'])),  mmfedEKTAcceDepth_testAcce['root_test'],
             color=color["cgen"],linestyle='dashed',
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Acce)")
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 0.8)
    ax2.grid()
    ax2.set_title(DATASET+" "+ "Acce-Depth")
    ax2.set_xlabel("#Global Rounds")

    ax3.plot(np.arange(len(mmfedavgRgbDepth_testRgb['root_test'])), mmfedavgRgbDepth_testRgb['root_test'],
             color=color["gen"],
             marker=marker["gen"], markevery=markers_on,
             label="Mm-FedAvg(Rgb)")
    ax3.plot(np.arange(len(mmfedavgRgbDepth_testDepth['root_test'])), mmfedavgRgbDepth_testDepth['root_test'],
             color=color["c"],linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="Mm-FedAvg(Depth)")
    ax3.plot(np.arange(len(mmfedEKTRgbDepth_testRgb['root_test'])), mmfedEKTRgbDepth_testRgb['root_test'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Rgb)")
    ax3.plot(np.arange(len(mmfedEKTRgbDepth_testDepth['root_test'])), mmfedEKTRgbDepth_testDepth['root_test'],
             color=color["cgen"],linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="FedMEKT(Depth)")
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 0.8)
    ax3.grid()
    ax3.set_title(DATASET + " " + "Rgb-Depth")
    ax3.set_xlabel("#Global Rounds")



    # handles, labels = ax1.get_legend_handles_labels()
    ax1.legend( loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    # handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    ax3.legend(loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET  + OUT_TYPE)

def plot_ur_fall_local():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15,4.4))

    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    mmfedavgAcceRgb_testAcce = read_data(RS_PATH + name['mmfedavg_acce_rgb_testacce'])
    mmfedavgAcceRgb_testRgb = read_data(RS_PATH + name['mmfedavg_acce_rgb_testrgb'])
    mmfedavgAcceDepth_testDepth = read_data(RS_PATH + name['mmfedavg_acce_depth_testdepth'])
    mmfedavgAcceDepth_testAcce = read_data(RS_PATH + name['mmfedavg_acce_depth_testacce'])
    mmfedavgRgbDepth_testRgb = read_data(RS_PATH + name['mmfedavg_rgb_depth_testrgb'])
    mmfedavgRgbDepth_testDepth = read_data(RS_PATH + name['mmfedavg_rgb_depth_testdepth'])
    mmfedEKTAcceRgb_testAcce = read_data(RS_PATH + name['mmfedEKT_acce_rgb_testacce'])
    mmfedEKTAcceRgb_testRgb = read_data(RS_PATH + name['mmfedEKT_acce_rgb_testrgb'])
    mmfedEKTAcceDepth_testDepth = read_data(RS_PATH + name['mmfedEKT_acce_depth_testdepth'])
    mmfedEKTAcceDepth_testAcce = read_data(RS_PATH + name['mmfedEKT_acce_depth_testacce'])
    mmfedEKTRgbDepth_testRgb = read_data(RS_PATH + name['mmfedEKT_rgb_depth_testrgb'])
    mmfedEKTRgbDepth_testDepth = read_data(RS_PATH + name['mmfedEKT_rgb_depth_testdepth'])


    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(mmfedavgAcceRgb_testAcce['avg_local_f1_acc'], label="Mm-FedAvg(Acce)",  color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(mmfedavgAcceRgb_testRgb['avg_local_f1_acc'])), mmfedavgAcceRgb_testRgb['avg_local_f1_acc'],
             color=color["c"], linestyle='dashdot',
             marker=marker["cgen"], markevery=markers_on,
             label="Mm-FedAvg(Rgb)")

    ax1.plot(np.arange(len(mmfedEKTAcceRgb_testAcce['avg_local_f1_acc'])), mmfedEKTAcceRgb_testAcce['avg_local_f1_acc'], color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Acce)")
    ax1.plot(np.arange(len(mmfedEKTAcceRgb_testRgb['avg_local_f1_acc'])), mmfedEKTAcceRgb_testRgb['avg_local_f1_acc'],
             color=color["cgen"], linestyle='dashdot',
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Rgb)")


    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 0.8)
    ax1.set_title(DATASET+" "+ "Acce-Rgb")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Local Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2


    # print("CDKTKL Global", f_data['root_test'][12])
    # print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])

    ax2.plot(np.arange(len( mmfedavgAcceDepth_testAcce['avg_local_f1_acc'])),  mmfedavgAcceDepth_testAcce['avg_local_f1_acc'],
             color=color["c"], linestyle='dashdot',
             marker=marker["cgen"], markevery=markers_on,
             label="Mm-FedAvg(Acce)")
    ax2.plot(np.arange(len(mmfedavgAcceDepth_testDepth['avg_local_f1_acc'])),
             mmfedavgAcceDepth_testDepth['avg_local_f1_acc'],
             color=color["gen"],
             marker=marker["gen"], markevery=markers_on,
             label="Mm-FedAvg(Depth)")

    ax2.plot(np.arange(len( mmfedEKTAcceDepth_testAcce['avg_local_f1_acc'])),  mmfedEKTAcceDepth_testAcce['avg_local_f1_acc'],
             color=color["cgen"],linestyle='dashdot',
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Acce)")
    ax2.plot(np.arange(len(mmfedEKTAcceDepth_testDepth['avg_local_f1_acc'])),
             mmfedEKTAcceDepth_testDepth['avg_local_f1_acc'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Depth)")
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 0.8)
    ax2.grid()
    ax2.set_title(DATASET+" "+ "Acce-Depth")
    ax2.set_xlabel("#Global Rounds")

    ax3.plot(np.arange(len(mmfedavgRgbDepth_testRgb['avg_local_f1_acc'])), mmfedavgRgbDepth_testRgb['avg_local_f1_acc'],
             color=color["gen"],
             marker=marker["gen"], markevery=markers_on,
             label="Mm-FedAvg(Rgb)")
    ax3.plot(np.arange(len(mmfedavgRgbDepth_testDepth['avg_local_f1_acc'])), mmfedavgRgbDepth_testDepth['avg_local_f1_acc'],
             color=color["c"],linestyle='dashdot',
             marker=marker["cgen"], markevery=markers_on,
             label="Mm-FedAvg(Depth)")
    ax3.plot(np.arange(len(mmfedEKTRgbDepth_testRgb['avg_local_f1_acc'])), mmfedEKTRgbDepth_testRgb['avg_local_f1_acc'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Rgb)")
    ax3.plot(np.arange(len(mmfedEKTRgbDepth_testDepth['avg_local_f1_acc'])), mmfedEKTRgbDepth_testDepth['avg_local_f1_acc'],
             color=color["cgen"],linestyle='dashdot',
             marker=marker["cgen"], markevery=markers_on,
             label="FedMEKT(Depth)")
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 0.8)
    ax3.grid()
    ax3.set_title(DATASET + " " + "Rgb-Depth")
    ax3.set_xlabel("#Global Rounds")



    # handles, labels = ax1.get_legend_handles_labels()
    ax1.legend( loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    # handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    ax3.legend(loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+"local perf"+DATASET  + OUT_TYPE)

def plot_ur_fall_project():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 4.4))

    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    mmfedavgAcceRgb_testAcce = read_data(RS_PATH + name['mmfedavg_acce_rgb_testacce'])
    mmfedavgAcceRgb_testRgb = read_data(RS_PATH + name['mmfedavg_acce_rgb_testrgb'])
    mmfedavgAcceDepth_testDepth = read_data(RS_PATH + name['mmfedavg_acce_depth_testdepth'])
    mmfedavgAcceDepth_testAcce = read_data(RS_PATH + name['mmfedavg_acce_depth_testacce'])
    mmfedavgRgbDepth_testRgb = read_data(RS_PATH + name['mmfedavg_rgb_depth_testrgb'])
    mmfedavgRgbDepth_testDepth = read_data(RS_PATH + name['mmfedavg_rgb_depth_testdepth'])
    mmfedEKDAcceRgb_testAcce = read_data(RS_PATH + name['FedEKD_acce_rgb_testacce'])
    mmfedEKDAcceRgb_testRgb = read_data(RS_PATH + name['FedEKD_acce_rgb_testrgb'])
    mmfedEKDAcceDepth_testDepth = read_data(RS_PATH + name['FedEKD_acce_depth_testdepth'])
    mmfedEKDAcceDepth_testAcce = read_data(RS_PATH + name['FedEKD_acce_depth_testacce'])
    mmfedEKDRgbDepth_testRgb = read_data(RS_PATH + name['FedEKD_rgb_depth_testrgb'])
    mmfedEKDRgbDepth_testDepth = read_data(RS_PATH + name['FedEKD_rgb_depth_testdepth'])


    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(mmfedavgAcceRgb_testAcce['root_test'], label="MM-FedAvg(Acce)",  color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(np.arange(len(mmfedavgAcceRgb_testRgb['root_test'])), mmfedavgAcceRgb_testRgb['root_test'],
             color=color["c"], linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="MM-FedAvg(Rgb)")

    ax1.plot(np.arange(len(mmfedEKDAcceRgb_testAcce['root_test'])), mmfedEKDAcceRgb_testAcce['root_test'], color=color["g"],
             marker=marker["cgen"], markevery=markers_on,
             label="MM-FedEKD(Acce)")
    ax1.plot(np.arange(len(mmfedEKDAcceRgb_testRgb['root_test'])), mmfedEKDAcceRgb_testRgb['root_test'],
             color=color["cgen"], linestyle='dashed',
             marker=marker["ggen"], markevery=markers_on,
             label="MM-FedEKD(Rgb)")



    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 0.8)
    ax1.set_title(DATASET+" "+ "Acce-Rgb")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2


    # print("CDKTKL Global", f_data['root_test'][12])
    # print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(np.arange(len( mmfedavgAcceDepth_testDepth['root_test'])),  mmfedavgAcceDepth_testDepth['root_test'],
             color=color["gen"],
             marker=marker["gen"], markevery=markers_on,
             label="MM-FedAvg(Depth)")
    ax2.plot(np.arange(len( mmfedavgAcceDepth_testAcce['root_test'])),  mmfedavgAcceDepth_testAcce['root_test'],
             color=color["c"], linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="MM-FedAvg(Acce)")
    ax2.plot(np.arange(len( mmfedEKDAcceDepth_testDepth['root_test'])),  mmfedEKDAcceDepth_testDepth['root_test'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="MM-FedEKD(Depth)")
    ax2.plot(np.arange(len( mmfedEKDAcceDepth_testAcce['root_test'])),  mmfedEKDAcceDepth_testAcce['root_test'],
             color=color["cgen"],linestyle='dashed',
             marker=marker["ggen"], markevery=markers_on,
             label="MM-FedEKD(Acce)")
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 0.8)
    ax2.grid()
    ax2.set_title(DATASET+" "+ "Acce-Depth")
    ax2.set_xlabel("#Global Rounds")

    ax3.plot(np.arange(len(mmfedavgRgbDepth_testRgb['root_test'])), mmfedavgRgbDepth_testRgb['root_test'],
             color=color["gen"],
             marker=marker["gen"], markevery=markers_on,
             label="MM-FedAvg(Rgb)")
    ax3.plot(np.arange(len(mmfedavgRgbDepth_testDepth['root_test'])), mmfedavgRgbDepth_testDepth['root_test'],
             color=color["c"],linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="MM-FedAvg(Depth)")
    ax3.plot(np.arange(len(mmfedEKDRgbDepth_testRgb['root_test'])), mmfedEKDRgbDepth_testRgb['root_test'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="MM-FedEKD(Rgb)")
    ax3.plot(np.arange(len(mmfedEKDRgbDepth_testDepth['root_test'])), mmfedEKDRgbDepth_testDepth['root_test'],
             color=color["cgen"],linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="MM-FedEKD(Depth)")
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 0.9)
    ax3.grid()
    ax3.set_title(DATASET + " " + "Rgb-Depth")
    ax3.set_xlabel("#Global Rounds")



    # handles, labels = ax1.get_legend_handles_labels()
    ax1.legend( loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    # handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    ax3.legend(loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET  + "Project"+ OUT_TYPE)
def plot_mhealth():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 4.4))

    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    mmfedavgAcceGyro_testAcce = read_data(RS_PATH + name['mmfedavg_acce_gyro_testacce'])
    mmfedavgAcceGyro_testGyro = read_data(RS_PATH + name['mmfedavg_acce_gyro_testgyro'])
    mmfedavgGyroMage_testGyro = read_data(RS_PATH + name['mmfedavg_gyro_mage_testgyro'])
    mmfedavgGyroMage_testMage = read_data(RS_PATH + name['mmfedavg_gyro_mage_testmage'])
    mmfedavgAcceMage_testAcce = read_data(RS_PATH + name['mmfedavg_acce_mage_testacce'])
    mmfedavgAcceMage_testMage = read_data(RS_PATH + name['mmfedavg_acce_mage_testmage'])
    mmfedEKTAcceGyro_testAcce = read_data(RS_PATH + name['mmfedEKT_acce_gyro_testacce'])
    mmfedEKTAcceGyro_testGyro  = read_data(RS_PATH + name['mmfedEKT_acce_gyro_testgyro'])
    mmfedEKTGyroMage_testGyro = read_data(RS_PATH + name['mmfedEKT_gyro_mage_testgyro'])
    mmfedEKTGyroMage_testMage = read_data(RS_PATH + name['mmfedEKT_gyro_mage_testmage'])
    mmfedEKTAcceMage_testAcce = read_data(RS_PATH + name['mmfedEKT_acce_mage_testacce'])
    mmfedEKTAcceMage_testMage = read_data(RS_PATH + name['mmfedEKT_acce_mage_testmage'])



    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(mmfedavgAcceGyro_testAcce['root_test'], label="Mm-FedAvg(Acce)",  color=color["gen"], marker=marker["gen"],
             markevery=markers_on)

    ax1.plot(np.arange(len(mmfedavgAcceGyro_testGyro['root_test'])), mmfedavgAcceGyro_testGyro['root_test'], color=color["c"],
             marker=marker["cgen"], markevery=markers_on,linestyle='dashdot',
             label="Mm-FedAvg(Gyro)")
    ax1.plot(np.arange(len( mmfedEKTAcceGyro_testAcce['root_test'])),  mmfedEKTAcceGyro_testAcce['root_test'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Acce)")
    ax1.plot(np.arange(len( mmfedEKTAcceGyro_testGyro['root_test'])),  mmfedEKTAcceGyro_testGyro['root_test'],
             color=color["cgen"],linestyle='dashdot',
             marker=marker["cgen"], markevery=markers_on,
             label="FedMEKT(Gyro)")

    ax1.set_xlim(0, XLim)
    ax1.set_ylim(0.4, 0.8)
    ax1.set_title(DATASET+" "+ "Acce-Gyro")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2


    # print("CDKTKL Global", f_data['root_test'][12])
    # print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(np.arange(len( mmfedavgGyroMage_testGyro['root_test'])),  mmfedavgGyroMage_testGyro['root_test'],
             color=color["gen"],
             marker=marker["gen"], markevery=markers_on,
             label="Mm-FedAvg(Gyro)")
    ax2.plot(np.arange(len( mmfedavgGyroMage_testMage['root_test'])),  mmfedavgGyroMage_testMage['root_test'],
             color=color["c"],linestyle='dashdot',
             marker=marker["cgen"], markevery=markers_on,
             label="Mm-FedAvg(Mage)")
    ax2.plot(np.arange(len( mmfedEKTGyroMage_testGyro['root_test'])),  mmfedEKTGyroMage_testGyro['root_test'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Gyro)")
    ax2.plot(np.arange(len( mmfedEKTGyroMage_testMage['root_test'])),  mmfedEKTGyroMage_testMage['root_test'],
             color=color["cgen"],linestyle='dashdot',
             marker=marker["cgen"], markevery=markers_on,
             label="FedMEKT(Mage)")
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(0.4, 0.8)
    ax2.grid()
    ax2.set_title(DATASET+" "+ "Gyro-Mage")
    ax2.set_xlabel("#Global Rounds")
    #
    ax3.plot(np.arange(len(mmfedavgAcceMage_testAcce['root_test'])), mmfedavgAcceMage_testAcce['root_test'],
             color=color["gen"],
             marker=marker["gen"], markevery=markers_on,
             label="Mm-FedAvg(Acce)")
    ax3.plot(np.arange(len(mmfedavgAcceMage_testMage['root_test'])), mmfedavgAcceMage_testMage['root_test'],
             color=color["c"],linestyle='dashdot',
             marker=marker["cgen"], markevery=markers_on,
             label="Mm-FedAvg(Mage)")
    ax3.plot(np.arange(len(mmfedEKTAcceMage_testAcce['root_test'])), mmfedEKTAcceMage_testAcce['root_test'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Acce)")
    ax3.plot(np.arange(len(mmfedEKTAcceMage_testMage['root_test'])), mmfedEKTAcceMage_testMage['root_test'],
             color=color["cgen"],linestyle='dashdot',
             marker=marker["cgen"], markevery=markers_on,
             label="FedMEKT(Mage)")
    ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(0.4, 0.8)
    ax3.grid()
    ax3.set_title(DATASET + " " + "Acce-Mage")
    ax3.set_xlabel("#Global Rounds")



    # handles, labels = ax1.get_legend_handles_labels()
    ax1.legend( loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    # handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    ax3.legend(loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET  + OUT_TYPE)


def plot_mhealth_local():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 4.4))

    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    mmfedavgAcceGyro_testAcce = read_data(RS_PATH + name['mmfedavg_acce_gyro_testacce'])
    mmfedavgAcceGyro_testGyro = read_data(RS_PATH + name['mmfedavg_acce_gyro_testgyro'])
    mmfedavgAcceMage_testAcce = read_data(RS_PATH + name['mmfedavg_acce_mage_testacce'])
    mmfedavgAcceMage_testMage = read_data(RS_PATH + name['mmfedavg_acce_mage_testmage'])
    mmfedavgGyroMage_testGyro = read_data(RS_PATH + name['mmfedavg_gyro_mage_testgyro'])
    mmfedavgGyroMage_testMage = read_data(RS_PATH + name['mmfedavg_gyro_mage_testmage'])
    mmfedEKTAcceGyro_testAcce = read_data(RS_PATH + name['mmfedEKT_acce_gyro_testacce'])
    mmfedEKTAcceGyro_testGyro  = read_data(RS_PATH + name['mmfedEKT_acce_gyro_testgyro'])
    mmfedEKTGyroMage_testGyro = read_data(RS_PATH + name['mmfedEKT_gyro_mage_testgyro'])
    mmfedEKTGyroMage_testMage = read_data(RS_PATH + name['mmfedEKT_gyro_mage_testmage'])
    mmfedEKTAcceMage_testAcce = read_data(RS_PATH + name['mmfedEKT_acce_mage_testacce'])
    mmfedEKTAcceMage_testMage = read_data(RS_PATH + name['mmfedEKT_acce_mage_testmage'])



    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(mmfedavgAcceGyro_testAcce['avg_local_f1_acc'], label="Mm-FedAvg(Acce)",  color=color["gen"], marker=marker["gen"],
             markevery=markers_on)

    ax1.plot(np.arange(len(mmfedavgAcceGyro_testGyro['avg_local_f1_acc'])), mmfedavgAcceGyro_testGyro['avg_local_f1_acc'], color=color["c"],
             marker=marker["cgen"], markevery=markers_on,linestyle='dashdot',
             label="Mm-FedAvg(Gyro)")
    ax1.plot(np.arange(len( mmfedEKTAcceGyro_testAcce['avg_local_f1_acc'])),  mmfedEKTAcceGyro_testAcce['avg_local_f1_acc'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Acce)")
    ax1.plot(np.arange(len( mmfedEKTAcceGyro_testGyro['avg_local_f1_acc'])),  mmfedEKTAcceGyro_testGyro['avg_local_f1_acc'],
             color=color["cgen"],linestyle='dashdot',
             marker=marker["cgen"], markevery=markers_on,
             label="FedMEKT(Gyro)")

    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title(DATASET+" "+ "Acce-Gyro")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Local Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2


    # print("CDKTKL Global", f_data['root_test'][12])
    # print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(np.arange(len( mmfedavgGyroMage_testGyro['avg_local_f1_acc'])),  mmfedavgGyroMage_testGyro['avg_local_f1_acc'],
             color=color["gen"],
             marker=marker["gen"], markevery=markers_on,
             label="Mm-FedAvg(Gyro)")
    ax2.plot(np.arange(len( mmfedavgGyroMage_testMage['avg_local_f1_acc'])),  mmfedavgGyroMage_testMage['avg_local_f1_acc'],
             color=color["c"],linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="Mm-FedAvg(Mage)")
    ax2.plot(np.arange(len( mmfedEKTGyroMage_testGyro['avg_local_f1_acc'])),  mmfedEKTGyroMage_testGyro['avg_local_f1_acc'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Gyro)")
    ax2.plot(np.arange(len( mmfedEKTGyroMage_testMage['avg_local_f1_acc'])),  mmfedEKTGyroMage_testMage['avg_local_f1_acc'],
             color=color["cgen"],linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="FedMEKT(Mage)")
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.grid()
    ax2.set_title(DATASET+" "+  "Gyro-Mage")
    ax2.set_xlabel("#Global Rounds")
    #
    ax3.plot(np.arange(len(mmfedavgAcceMage_testAcce['avg_local_f1_acc'])), mmfedavgAcceMage_testAcce['avg_local_f1_acc'],
             color=color["gen"],
             marker=marker["gen"], markevery=markers_on,
             label="Mm-FedAvg(Acce)")
    ax3.plot(np.arange(len(mmfedavgAcceMage_testMage['avg_local_f1_acc'])), mmfedavgAcceMage_testMage['avg_local_f1_acc'],
             color=color["c"],linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="Mm-FedAvg(Mage)")
    ax3.plot(np.arange(len(mmfedEKTAcceMage_testAcce['avg_local_f1_acc'])), mmfedEKTAcceMage_testAcce['avg_local_f1_acc'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Acce)")
    ax3.plot(np.arange(len(mmfedEKTAcceMage_testMage['avg_local_f1_acc'])), mmfedEKTAcceMage_testMage['avg_local_f1_acc'],
             color=color["cgen"],linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="FedMEKT(Mage)")
    ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.grid()
    ax3.set_title(DATASET + " " + "Acce-Mage")
    ax3.set_xlabel("#Global Rounds")



    # handles, labels = ax1.get_legend_handles_labels()
    ax1.legend( loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    # handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    ax3.legend(loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ "Local perf"+DATASET  + OUT_TYPE)
def plot_opp():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5.4,4.4))

    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    mmfedavgAcceGyro_testAcce = read_data(RS_PATH + name['mmfedavg_acce_gyro_testacce'])
    mmfedavgAcceGyro_testGyro = read_data(RS_PATH + name['mmfedavg_acce_gyro_testgyro'])
    mmfedEKTAcceGyro_testAcce = read_data(RS_PATH + name['mmfedEKT_acce_gyro_testacce'])
    mmfedEKTAcceGyro_testGyro = read_data(RS_PATH + name['mmfedEKT_acce_gyro_testgyro'])




    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(mmfedavgAcceGyro_testAcce['root_test'], label="Mm-FedAvg(Acce)",  color=color["gen"], marker=marker["gen"],
             markevery=markers_on)

    ax1.plot(np.arange(len(mmfedavgAcceGyro_testGyro['root_test'])), mmfedavgAcceGyro_testGyro['root_test'], color=color["c"],
             marker=marker["cgen"], markevery=markers_on,linestyle='dashdot',
             label="Mm-FedAvg(Gyro)")
    ax1.plot(np.arange(len(mmfedEKTAcceGyro_testAcce['root_test'])), mmfedEKTAcceGyro_testAcce['root_test'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Acce)")
    ax1.plot(np.arange(len(mmfedEKTAcceGyro_testGyro['root_test'])), mmfedEKTAcceGyro_testGyro['root_test'],
             color=color["cgen"],linestyle='dashdot',
             marker=marker["cgen"], markevery=markers_on,
             label="FedMEKT(Gyro)")


    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 0.8)
    ax1.set_title(DATASET+" "+ "Acce-Gyro")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2







    # handles, labels = ax1.get_legend_handles_labels()
    ax1.legend( loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    # handles2, labels2 = ax2.get_legend_handles_labels()

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET  + OUT_TYPE)

def plot_opp_local():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5.4,4.4))

    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    mmfedavgAcceGyro_testAcce = read_data(RS_PATH + name['mmfedavg_acce_gyro_testacce'])
    mmfedavgAcceGyro_testGyro = read_data(RS_PATH + name['mmfedavg_acce_gyro_testgyro'])
    mmfedEKTAcceGyro_testAcce = read_data(RS_PATH + name['mmfedEKT_acce_gyro_testacce'])
    mmfedEKTAcceGyro_testGyro = read_data(RS_PATH + name['mmfedEKT_acce_gyro_testgyro'])




    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(mmfedavgAcceGyro_testAcce['avg_local_f1_acc'], label="Mm-FedAvg(Acce)",  color=color["gen"], marker=marker["gen"],
             markevery=markers_on)

    ax1.plot(np.arange(len(mmfedavgAcceGyro_testGyro['avg_local_f1_acc'])), mmfedavgAcceGyro_testGyro['avg_local_f1_acc'], color=color["c"],
             linestyle='dashed',marker=marker["cgen"], markevery=markers_on,
             label="Mm-FedAvg(Gyro)")
    ax1.plot(np.arange(len(mmfedEKTAcceGyro_testAcce['avg_local_f1_acc'])), mmfedEKTAcceGyro_testAcce['avg_local_f1_acc'],
             color=color["g"],
             marker=marker["ggen"], markevery=markers_on,
             label="FedMEKT(Acce)")
    ax1.plot(np.arange(len(mmfedEKTAcceGyro_testGyro['avg_local_f1_acc'])), mmfedEKTAcceGyro_testGyro['avg_local_f1_acc'],
             color=color["cgen"],linestyle='dashed',
             marker=marker["cgen"], markevery=markers_on,
             label="FedMEKT(Gyro)")


    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title(DATASET+"  "+ "Acce-Gyro")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Local Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2







    # handles, labels = ax1.get_legend_handles_labels()
    ax1.legend( loc="lower right",
               prop={'size': 14})  # mode="expand",  mode="expand", frameon=False,
    # handles2, labels2 = ax2.get_legend_handles_labels()

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH +"local perf"+DATASET  + OUT_TYPE)

def plot_mhealth_gyro_mage():
    plt.rcParams.update({'font.size': 60})
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(45, 25))

    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15.0, 4.4))
    mmfedavgGyroMage_testGyro = read_data(RS_PATH + name['mmfedavg_gyro_mage_testgyro'])
    mmfedavgGyroMage_testMage = read_data(RS_PATH + name['mmfedavg_gyro_mage_testmage'])
    mmfedEKTGyroMage_testGyro_cos =read_data(RS_PATH + name['mmfedEKT_gyro_mage_testgyro_cos'])
    mmfedEKTGyroMage_testMage_cos =read_data(RS_PATH + name['mmfedEKT_gyro_mage_testmage_cos'])
    mmfedEKTGyroMage_testGyro_con = read_data(RS_PATH + name['mmfedEKT_gyro_mage_testgyro_con'])
    mmfedEKTGyroMage_testMage_con = read_data(RS_PATH + name['mmfedEKT_gyro_mage_testmage_con'])


    # print("CDKTKLN Global :", f_data['root_test'][12])
    # print("CDKTKLN C-GEN:", f_data['cg_avg_data_test'][XLim - 1])

    ax1.plot(mmfedavgGyroMage_testGyro['root_test'], label="mmFedAvg",  color=color["gen"], marker=marker["gen"],
             markevery=markers_on)

    ax1.plot(np.arange(len(mmfedEKTGyroMage_testGyro_cos['root_test'])), mmfedEKTGyroMage_testGyro_cos['root_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="mmFedEKT_Cos")

    ax1.plot(np.arange(len(mmfedEKTGyroMage_testGyro_con['root_test'])), mmfedEKTGyroMage_testGyro_con['root_test'],
             color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on,
             label="mmFedEKT_Con")

    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title(DATASET+" "+ "Gyro-Mage"+"Test_Gyro")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2


    # print("CDKTKL Global", f_data['root_test'][12])
    # print("CDKTKL C-SPE:", f_data['cs_avg_data_test'][XLim - 1])
    ax2.plot(np.arange(len( mmfedavgGyroMage_testMage['root_test'])),  mmfedavgGyroMage_testMage['root_test'],
             color=color["gen"],
             marker=marker["gen"], markevery=markers_on,
             label="mmFedAvg")
    ax2.plot(np.arange(len( mmfedEKTGyroMage_testMage_cos['root_test'])),  mmfedEKTGyroMage_testMage_cos['root_test'],
             color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="mmFedEKT_Cos")
    ax2.plot(np.arange(len( mmfedEKTGyroMage_testMage_con['root_test'])),  mmfedEKTGyroMage_testMage_con['root_test'],
             color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on,
             label="mmFedEKT_Con")
    # ax3.legend(loc="best", prop={'size': 8})
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.grid()
    ax2.set_title(DATASET+" "+ "Gyro-Mage"+"Test_Mage")
    ax2.set_xlabel("#Global Rounds")




    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=3,
               prop={'size': 60})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH+ DATASET + "Gyro-Mage" + OUT_TYPE)
def plot_demlearn_vs_demlearn_p():
    plt.rcParams.update({'font.size': 16})
    fig, (ax3, ax2, ax6, ax5) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['avg3wf'])
    ax2.plot(f_data['root_test'], label="GEN", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax2.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax2.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax2.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax2.set_xlim(0, XLim)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Testing Accuracy")
    # ax2.set_title("DemLearn: $K=4$, Fixed")
    ax2.set_title("DemLearn: Fixed")
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()

    f_data = read_data(RS_PATH + name['avg3w'])
    ax3.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax3.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax3.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax3.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.set_title("DemLearn")
    ax3.set_xlabel("#Global Rounds")
    ax3.grid()

    f_data = read_data(RS_PATH + name['prox3wf'])
    ax5.plot(f_data['root_test'], label="GEN", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax5.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax5.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax5.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax5.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax5.set_xlim(0, XLim)
    ax5.set_ylim(0, 1)
    # ax5.set_title("DemLearn-P: $K=4$, Fixed")
    ax5.set_xlabel("#Global Rounds")
    ax5.set_title("DemLearn-P: Fixed")
    ax5.set_xlabel("#Global Rounds")
    ax5.grid()
    f_data = read_data(RS_PATH + name['prox3w'])
    ax6.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax6.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax6.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax6.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax6.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax6.set_xlim(0, XLim)
    ax6.set_ylim(YLim, 1)
    # ax6.set_title("DemLearn-P: $K=4$")
    ax6.set_title("DemLearn-P")
    ax6.set_xlabel("#Global Rounds")
    ax6.grid()


    plt.tight_layout()
    # plt.grid(linewidth=0.25)

    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1,  ncol=5, prop={'size': 14})  # mode="expand",mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_K_vary"+OUT_TYPE)
    return 0

def plot_demlearn_p_mu_vari():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax4, ax3) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['prox3wmu005'])
    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax1.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    # ax1.legend(loc="best", prop={'size': 8})
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("DemLearn-P: $\mu=0.005$")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2
    f_data = read_data(RS_PATH + name['prox3wmu002'])

    ax2.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax2.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax2.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax2.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.set_title("DemLearn-P: $\mu=0.002$")
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()

    f_data = read_data(RS_PATH + name['prox3wmu0005'])
    ax3.plot(f_data['root_test'], label="Generalization", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(f_data['gs_level_test'][-2, :, 0], label="Group-Generalization", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax3.plot(f_data['gg_level_test'][-2, :, 0], label="Group-Specialization", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax3.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="Client-Specialization")
    ax3.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="Client-Generalization")
    # ax1.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.set_title("DemLearn-P: $\mu=0.0005$")
    ax3.set_xlabel("#Global Rounds")
    #ax3.set_ylabel("Testing Accuracy")
    ax3.grid()
    # subfig1-end---begin---subfig 2
    f_data = read_data(RS_PATH + name['prox3wmu001'])

    ax4.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax4.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax4.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax4.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.set_title("DemLearn-P: $\mu=0.001$")
    ax4.set_xlabel("#Global Rounds")
    ax4.grid()
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5, prop={'size': 15})
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    # plt.grid(linewidth=0.25)

    plt.savefig(PLOT_PATH+ DATASET + "_dem_prox_mu_vary"+OUT_TYPE)
    return 0


def plot_demlearn_gamma_vari():
    plt.rcParams.update({'font.size': 14})
    fig, (ax3, ax2, ax1) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10.0, 4.2))
    # fig, (ax3, ax2, ax1, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(13.0, 4.2))
    f_data = read_data(RS_PATH + name['avg3w'])
    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax1.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    # ax1.legend(loc="best", prop={'size': 8})
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("DemLearn: $\gamma=0.6$")
    ax1.set_xlabel("#Global Rounds")
    # ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2
    f_data = read_data(RS_PATH + name['avg3wg08'])

    ax2.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax2.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax2.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax2.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.set_title("DemLearn: $\gamma=0.8$")
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()

    f_data = read_data(RS_PATH + name['avg3g1'])
    ax3.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(f_data['gs_level_test'][-2, :, 0], label="Group-Generalization", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax3.plot(f_data['gg_level_test'][-2, :, 0], label="Group-Specialization", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax3.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="Client-Specialization")
    ax3.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="Client-Generalization")
    # ax1.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.set_title("DemLearn: $\gamma=1.0$")
    ax3.set_xlabel("#Global Rounds")
    ax3.set_ylabel("Testing Accuracy")
    ax3.grid()

    plt.tight_layout()
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 14})  # mode="expand",
    plt.subplots_adjust(bottom=0.24)
    plt.savefig(PLOT_PATH+ DATASET+"_dem_avg_gamma_vary"+OUT_TYPE)
    return 0


def plot_demlearn_gamma_vari_clients():
    plt.rcParams.update({'font.size': 14})
    fig, (ax3, ax2, ax1) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10.0, 3.96))
    # fig, (ax3, ax2, ax1, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(13.0, 4.2))
    f_data = read_data(RS_PATH + name['avg3w'])
    ax1.plot(f_data['cs_data_test'], linewidth=1.2)
    ax1.plot(f_data['root_test'], linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)

    # ax1.legend(loc="best", prop={'size': 8})
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(0.6, 1.01)
    ax1.set_title("Client-Spec: $\gamma=0.6$")
    ax1.set_xlabel("#Global Rounds")
    # ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2
    f_data = read_data(RS_PATH + name['avg3wg08'])
    ax2.plot(f_data['cs_data_test'], linewidth=1.2)
    ax2.plot(f_data['root_test'], linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)

    ax2.set_xlim(0, XLim)
    ax2.set_ylim(0.6, 1.01)
    ax2.set_title("Client-Spec: $\gamma=0.8$")
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()

    f_data = read_data(RS_PATH + name['avg3g1'])
    ax3.plot(f_data['cs_data_test'], linewidth=1.2)
    ax3.plot(f_data['root_test'], linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)

    # ax1.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(0.6, 1.01)
    ax3.set_title("Client-Spec: $\gamma=1.0$")
    ax3.set_xlabel("#Global Rounds")
    ax3.set_ylabel("Testing Accuracy")
    ax3.grid()


    plt.tight_layout()
    plt.savefig(PLOT_PATH+ DATASET+"_dem_avg_gamma_vary_clients"+OUT_TYPE)
    return 0

def plot_demlearn_w_vs_g():
    plt.rcParams.update({'font.size': 12})
    # plt.grid(linewidth=0.25)
    # fig, ((ax1, ax2, ax3),(ax4, ax5, ax6))= plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10.0, 7))
    fig, (ax1, ax4, ax2, ax3) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(13.0, 4.2))
    f_data = read_data(RS_PATH + name['avg3w'])
    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax1.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    # ax1.legend(loc="best", prop={'size': 8})
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("DemLearn:W-Clustering")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    ax2.plot(f_data['cg_data_test'], linewidth=0.7)
    ax2.plot(f_data['root_test'], linestyle="--", color=color["gen"], marker=marker["gen"], linewidth=2,
             markevery=markers_on)
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.set_title("Client-GEN: W-Clustering")

    f_data = read_data(RS_PATH + name['avg3g'])
    ax4.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax4.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax4.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax4.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.set_title("DemLearn: G-Clustering")
    ax4.set_xlabel("#Global Rounds")
    # ax4.set_ylabel("Testing Accuracy")
    ax4.grid()

    ax3.plot(f_data['cg_data_test'], linewidth=0.7)
    ax3.plot(f_data['root_test'], linestyle="--", color=color["gen"], marker=marker["gen"], linewidth=2,
             markevery=markers_on)
    ax3.set_xlabel("#Global Rounds")
    ax3.grid()
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.set_title("Client-GEN: G-Clustering")

    plt.tight_layout()

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1,  ncol=5, prop={'size': 14})  # mode="expand",mode="expand", frameon=False,

    plt.subplots_adjust(bottom=0.24)
    plt.savefig(PLOT_PATH + DATASET + "_w_vs_g"+OUT_TYPE)
    return 0

def plot_all_figs():
    # global CLUSTER_METHOD
    # plot_dem_vs_fed()  # plot comparision FED vs DEM
    # plot_demlearn_vs_demlearn_p()  # DEM, PROX vs K level
    # plot_demlearn_p_mu_vari()  # DEM Prox vs mu vary
    plot_fixed_users()
    plot_subset_users()
    plot_metric_fixed()
    plot_metric_subset()
    plot_same_vs_hetero()
    # ### SUPPLEMENTAL FIGS ####
    # plot_demlearn_gamma_vari() # DEM AVG vs Gamma vary
    # plot_demlearn_gamma_vari_clients()
    # plot_demlearn_w_vs_g()
    # # ##-------DENDOGRAM PLOT --------------------------------------------------------------------------------#
    # CLUSTER_METHOD = "gradient"
    # # plot_dendo_data_dem(file_name="prox3g")  # change file_name in order to get correct file to plot   #|
    # plot_dendo_data_dem(file_name="avg3g")  # change file_name in order to get correct file to plot   #|
    # plot_dendo_data_dem_shashi(file_name="avg3g", type="Gradient")
    # CLUSTER_METHOD = "weight"
    # # plot_dendo_data_dem(file_name="prox3w")
    # plot_dendo_data_dem(file_name="avg3w")
    # plot_dendo_data_dem_shashi(file_name="avg3w", type="Weight")
    ##-------DENDOGRAM PLOT --------------------------------------------------------------------------------#
    # plot_from_file()
    # plt.show()


if __name__=='__main__':
    PLOT_PATH = PLOT_PATH_FIG
    RS_PATH = FIG_PATH


    print("----- PLOT MHEALTH ------")
    DATASET = "mHealth"
    NUM_GLOBAL_ITERS = 100
    XLim = NUM_GLOBAL_ITERS
    Den_GAP = 7
    name = {

        "mmfedavg_acce_gyro_testacce":"mmFedAvg_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.009_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch1.h5",
        "mmfedavg_acce_gyro_testgyro":"mmFedAvg_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.009_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch1.h5",
        "mmfedavg_acce_mage_testacce": "mmFedAvg_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.05_eta0.05_beta0.009_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch1.h5",
        "mmfedavg_acce_mage_testmage": "mmFedAvg_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.05_eta0.05_beta0.009_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch1.h5",
        "mmfedavg_gyro_mage_testgyro": "mmFedAvg_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.05_eta0.05_beta0.009_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch1.h5",
        "mmfedavg_gyro_mage_testmage": "mmFedAvg_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.05_eta0.05_beta0.009_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch1.h5",
        "mmfedEKT_acce_gyro_testacce": "mmFedEKT_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse.h5",
        "mmfedEKT_acce_gyro_testgyro": "mmFedEKT_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.04_eta0.04_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsFalse.h5",
        "mmfedEKT_gyro_mage_testgyro":"mmFedEKT_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsFalse.h5",
        "mmfedEKT_gyro_mage_testmage":"mmFedEKT_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.06_gamma0.06_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsFalse.h5",
        "mmfedEKT_acce_mage_testacce":"mmFedEKT_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5",
        "mmfedEKT_acce_mage_testmage":"mmFedEKT_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.08_gamma0.08_SSTrue_gmKL_lmKL_ratio0.11_depoch3.h5",
        "mmfedEKT_acce_gyro_testacce_gcls":"mmFedEKT_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsTrue.h5",
        "mmfedEKT_acce_gyro_testgyro_gcls":"mmFedEKT_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.03_eta0.03_beta0.006_gamma0.006_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsTrue.h5",
        "mmfedEKT_gyro_mage_testgyro_gcls":"mmFedEKT_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsTrue.h5",
        "mmfedEKT_gyro_mage_testmage_gcls":"mmFedEKT_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.06_gamma0.06_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5",
        "mmfedEKT_acce_gyro_testacce_onelayer": "mmFedEKT_mhealth_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerTrue_globalclsFalse.h5",
        "mmfedEKT_acce_gyro_testgyro_onelayer": "mmFedEKT_mhealth_LAB_TB_I100_Maacce_Mbgyro_alpha0.03_eta0.03_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerTrue_globalclsFalse.h5",
        "mmfedEKT_gyro_mage_testgyro_onelayer": "mmFedEKT_mhealth_LAB_TA_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerTrue_globalclsFalse.h5",
        "mmfedEKT_gyro_mage_testmage_onelayer": "mmFedEKT_mhealth_LAB_TB_I100_Magyro_Mbmage_alpha0.07_eta0.07_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5",
        "mmfedEKT_acce_mage_testacce_onelayer": "mmFedEKT_mhealth_LAB_TA_I100_Maacce_Mbmage_alpha0.1_eta0.1_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerTrue_globalclsFalse.h5",
        "mmfedEKT_acce_mage_testmage_onelayer": "mmFedEKT_mhealth_LAB_TB_I100_Maacce_Mbmage_alpha0.08_eta0.08_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch3_onelayerTrue_globalclsFalse.h5"

    }


    # plot_mhealth()
    # plot_mhealth_local()
    # plot_all_figs()
    # plot_KCC2022()
    # plot_mhealth_acce_gyro()
    # plot_mhealth_acce_mage()
    # plot_mhealth_gyro_mage()
    # plot_fixed_users()
    # plot_subset_users()
    # plot_same_vs_hetero()
    # plot_alpha_effect2()
    # plot_beta_effect2()
    # plot_KSC()
    print("----- PLOT OPP ------")
    DATASET = "Opp"
    NUM_GLOBAL_ITERS = 100
    XLim = NUM_GLOBAL_ITERS
    Den_GAP = 7
    name = {

        "mmfedavg_acce_gyro_testacce":"mmFedAvg_opp_LAB_TA_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5",
        "mmfedavg_acce_gyro_testgyro":"mmFedAvg_opp_LAB_TB_I100_Maacce_Mbgyro_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5",
        "mmfedEKT_acce_gyro_testacce":"mmFedEKT_opp_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.009_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsDrFalse.h5",
        "mmfedEKT_acce_gyro_testgyro":"mmFedEKT_opp_LAB_TB_I100_Maacce_Mbgyro_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse.h5",
        "mmfedEKT_acce_gyro_testacce_gcls":"mmFedEKT_opp_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.009_gamma0.009_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsTrue.h5",
        "mmfedEKT_acce_gyro_testgyro_gcls":"mmFedEKT_opp_LAB_TB_I100_Maacce_Mbgyro_alpha0.01_eta0.01_beta0.007_gamma0.007_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerFalse_globalclsTrue.h5",
        "mmfedEKT_acce_gyro_testacce_onelayer": "mmFedEKT_opp_LAB_TA_I100_Maacce_Mbgyro_alpha0.05_eta0.05_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerTrue_globalclsFalse.h5",
        "mmfedEKT_acce_gyro_testgyro_onelayer": "mmFedEKT_opp_LAB_TB_I100_Maacce_Mbgyro_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch1_onelayerTrue_globalclsFalse.h5"


    }
    # plot_opp()
    # plot_opp_local()
    print("----- PLOT URFall ------")
    DATASET = "URFall"
    NUM_GLOBAL_ITERS = 100
    XLim = NUM_GLOBAL_ITERS
    Den_GAP = 7
    name = {

        "mmfedavg_acce_rgb_testacce":"mmFedAvg_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5",
        "mmfedavg_acce_rgb_testrgb":"mmFedAvg_ur_fall_LAB_TB_I100_Maacce_Mbrgb_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsFalse.h5",
        "mmfedavg_acce_depth_testdepth":"mmFedAvg_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5",
        "mmfedavg_acce_depth_testacce":"mmFedAvg_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5",
        "mmfedavg_rgb_depth_testrgb":"mmFedAvg_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.05_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5",
        "mmfedavg_rgb_depth_testdepth":"mmFedAvg_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.05_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5",
        "mmfedEKT_acce_rgb_testacce":"mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse.h5",
        "mmfedEKT_acce_rgb_testrgb": "mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbrgb_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsFalse.h5",
        "mmfedEKT_acce_depth_testdepth":"mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse.h5",
        "mmfedEKT_acce_depth_testacce":"mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse.h5",
        "mmfedEKT_rgb_depth_testrgb":"mmFedEKT_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.1_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrFalse.h5",
        "mmfedEKT_rgb_depth_testdepth":"mmFedEKT_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2.h5",
        "mmfedEKT_acce_rgb_testacce_gcls":"mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.07_eta0.07_beta0.03_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5",
        "mmfedEKT_acce_depth_testacce_gcls":"mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.08_eta0.08_beta0.07_gamma0.07_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5",
        "mmfedEKT_acce_depth_testdepth_gcls":"mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.07_eta0.07_beta0.05_gamma0.05_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5",
        "mmfedEKT_rgb_depth_testrgb_gcls":"mmFedEKT_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.1_eta0.05_beta0.04_gamma0.04_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5",
        "mmfedEKT_rgb_depth_testdepth_gcls":"mmFedEKT_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.02_eta0.02_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsTrue.h5",
        "mmfedEKT_acce_depth_testacce_onelayer": "mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha0.08_eta0.08_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5",
        "mmfedEKT_acce_depth_testdepth_onelayer": "mmFedEKT_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha0.07_eta0.07_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5",
        "mmfedEKT_rgb_depth_testrgb_onelayer": "mmFedEKT_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5",
        "mmfedEKT_rgb_depth_testdepth_onelayer": "mmFedEKT_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha0.1_eta0.1_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5",
        "mmfedEKT_acce_rgb_testacce_onelayer": "mmFedEKT_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha0.1_eta0.1_beta0.005_gamma0.005_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerTrue_globalclsFalse.h5",
        "FedEKD_acce_rgb_testacce": "FedEKD_ur_fall_LAB_TA_I100_Maacce_Mbrgb_alpha3_eta0.02_beta9_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5",
        "FedEKD_acce_rgb_testrgb": "FedEKD_ur_fall_LAB_TB_I100_Maacce_Mbrgb_alpha3_eta0.02_beta5_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5",
        "FedEKD_acce_depth_testdepth": "FedEKD_ur_fall_LAB_TB_I100_Maacce_Mbdepth_alpha3_eta0.02_beta5_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5",
        "FedEKD_acce_depth_testacce": "FedEKD_ur_fall_LAB_TA_I100_Maacce_Mbdepth_alpha3_eta0.02_beta5_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5",
        "FedEKD_rgb_depth_testrgb": "FedEKD_ur_fall_LAB_TA_I100_Margb_Mbdepth_alpha3_eta0.02_beta1_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5",
        "FedEKD_rgb_depth_testdepth": "FedEKD_ur_fall_LAB_TB_I100_Margb_Mbdepth_alpha3_eta0.02_beta3_gamma0.03_SSTrue_gmKL_lmKL_ratio0.11_depoch2_onelayerFalse_globalclsDrTrue_publicratio1.h5",

    }
    # plot_ur_fall()
    # plot_ur_fall_local()
    plot_ur_fall_project()
    # plot_all_figs()
    # plot_KCC2022()
    # plot_fixed_users()
    # plot_subset_users()
    # plot_same_vs_hetero()
    # plot_alpha_effect2()
    # plot_beta_effect2()
    # plot_KSC()
    plt.show()

