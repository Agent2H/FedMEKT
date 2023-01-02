import h5py as hf
import numpy as np
from Setting import *
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

plt.rcParams.update({'font.size': 16})  #font size 10 12 14 16 main 16
plt.rcParams['lines.linewidth'] = 2
XLim=100
YLim=0.1
#Global variable
# markers_on = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] #maker only at x = markers_on[]
markers_on = 10
OUT_TYPE = ".pdf" #.eps or .pdf #output figure type

color = {
    "gen": "royalblue",
    "cspe": "forestgreen",
    "cgen": "red",
    "c": "cyan",
    "gspe": "darkorange",  #magenta
    "gg": "yellow",
    "ggen": "darkviolet",
    "w": "white"
}
marker = {
    "gen": "8",
    "gspe": "s",
    "ggen": "P",
    "cspe": "p",
    "cgen": "*"
}

def  write_file(file_name = "../results/untitled.h5", **kwargs):
    with hf.File(file_name, "w") as data_file:
        for key, value in kwargs.items():
            #print("%s == %s" % (key, value))
            data_file.create_dataset(key, data=value)
    print("Successfully save to file!")

def read_data(file_name = "../results/untitled.h5"):
    print(":/")
    dic_data = {}
    with hf.File(file_name, "r") as f:
        # List all groups
        #print("Keys: %s" % f.keys())
        for key in f.keys():
            dic_data[key] = f[key][:]
    return  dic_data



def plot_dendrogram(rs_linkage_matrix, round, alg):
    # Plot the corresponding dendrogram
    # change p value to 5 if we want to get 5 levels
    #dendogram supporting plot inside buitlin function, so we dont need to create our own method to plot figure or results
    plt.title('#Round=%s'%(round))

    rs_dendrogram = dendrogram(rs_linkage_matrix, truncate_mode='level', p=K_Levels)
    if(MODEL_TYPE == "cnn"):
        if(CLUSTER_METHOD == "gradient"):
            plt.ylim(0, 0.0006)
        else:
            plt.ylim(0, 1.)
    else:
        plt.ylim(0,1.5)

def plot_dendo_data_dem():
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(6,5))
    f_data = read_data(rs_file_path)
    dendo_data = f_data['dendo_data']
    dendo_data_round = f_data['dendo_data_round']
    print(dendo_data_round)
    i = 0
    t = 0
    while( t < NUM_GLOBAL_ITERS):
        plt.clf()
        plot_dendrogram(dendo_data[i], dendo_data_round[i], RUNNING_ALG)
        plt.tight_layout()
        plt.savefig(PLOT_PATH + "den_I" + str(t) + OUT_TYPE)
        t+= TREE_UPDATE_PERIOD
        i+=1

    return 0

def plot_from_file():
    f_data = read_data(rs_file_path)
    # if("dem" in RUNNING_ALG):
    #     ### PLOT DENDROGRAM ####
    #     plot_dendo_data_dem()
    #     # dendo_data = f_data['dendo_data']
    #     # dendo_data_round = f_data['dendo_data_round']
    #     # i=0
    #     # for m_linkage in dendo_data:
    #     #     plot_dendrogram(m_linkage, dendo_data_round[i], RUNNING_ALG)
    #     #     i+=1
    print("DEM-AI --------->>>>> Plotting")
    print("Algorithm:",RUNNING_ALG)
    alg_name = RUNNING_ALG+ "_"

    plt.figure(4)
    plt.clf()
    plt.plot(f_data['root_test'], label="Root_test", linestyle="--")
    if("dem" in RUNNING_ALG):
        # for k in range (K_Levels):
            plt.plot(f_data['gs_level_test'][-2,:,0], label="Gr(K)_spec_test", linestyle="-.")
            plt.plot(f_data['gg_level_test'][-2,:,0], label="Gr(K)_gen_test", linestyle="-.")


    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], label="Client_spec_test")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], label="Client_gen_test")
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(YLim, 1.02)
    plt.grid()
    plt.title("AVG Clients Model (Spec-Gen) Testing Accuracy")
    plt.savefig(PLOT_PATH + alg_name+"AVGC_Spec_Gen_Testing.pdf")

    plt.figure(7)
    plt.clf()
    plt.plot(f_data['root_test'], linestyle="--", label="root test")
    plt.plot(f_data['cs_data_test'])
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(0, 1.02)
    plt.grid()
    plt.title("Testing Client Specialization")
    plt.savefig(PLOT_PATH + alg_name + "C_Spec_Testing.pdf")

    plt.figure(8)
    plt.clf()
    plt.plot(f_data['root_train'], linestyle="--", label="root train")
    plt.plot(f_data['cs_data_train'])
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(0, 1.02)
    plt.grid()
    plt.title("Training Client Specialization")
    plt.savefig(PLOT_PATH + alg_name + "C_Spec_Training.pdf")

    plt.figure(9)
    plt.clf()
    plt.plot(f_data['cg_data_test'])
    plt.plot(f_data['root_test'], linestyle="--", label="root test")
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(0, 1.02)
    plt.grid()
    plt.title("Testing Client Generalization")
    plt.savefig(PLOT_PATH + alg_name + "C_Gen_Testing.pdf")

    # plt.figure(10)
    # plt.clf()
    # plt.plot(f_data['cg_data_train'])
    # plt.plot(f_data['root_train'], linestyle="--", label="root train")
    # plt.legend()
    # plt.xlabel("Global Rounds")
    # plt.ylim(0, 1.02)
    # plt.grid()
    # plt.title("Training Client Generalization")
    # plt.savefig(PLOT_PATH + alg_name + "C_Gen_Training.pdf")

    plt.show()

    # print("** Summary Results: ---- Training ----")
    # print("AVG Clients Specialization - Training:", f_data['cs_avg_data_train'])
    # print("AVG Clients Generalization - Training::", f_data['cg_avg_data_train'])
    # print("Root performance - Training:", f_data['root_train'])
    # print("** Summary Results: ---- Testing ----")
    # print("AVG Clients Specialization - Testing:", f_data['cs_avg_data_test'])
    # print("AVG Clients Generalization - Testing:", f_data['cg_avg_data_test'])
    # print("Root performance - Testing:", f_data['root_test'])
def plot_from_file2():
    f_data = read_data(rs_file_path)
    # if("dem" in RUNNING_ALG):
    #     ### PLOT DENDROGRAM ####
    #     plot_dendo_data_dem()
    #     # dendo_data = f_data['dendo_data']
    #     # dendo_data_round = f_data['dendo_data_round']
    #     # i=0
    #     # for m_linkage in dendo_data:
    #     #     plot_dendrogram(m_linkage, dendo_data_round[i], RUNNING_ALG)
    #     #     i+=1
    print("DEM-AI --------->>>>> Plotting")
    print("Algorithm:",RUNNING_ALG)
    alg_name = RUNNING_ALG+ "_"

    plt.figure(4)
    plt.clf()
    plt.plot(f_data['root_test'], label="Root_test", linestyle="--")
    if("dem" in RUNNING_ALG):
        # for k in range (K_Levels):
            plt.plot(f_data['gs_level_test'][-2,:,0], label="Gr(K)_spec_test", linestyle="-.")
            plt.plot(f_data['gg_level_test'][-2,:,0], label="Gr(K)_gen_test", linestyle="-.")

    #
    # plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], label="Client_spec_test")
    # plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], label="Client_gen_test")
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(YLim, 1.02)
    plt.grid()
    plt.title("AVG Clients Model (Spec-Gen) Testing Accuracy")
    plt.savefig(PLOT_PATH + alg_name+"AVGC_Spec_Gen_Testing.pdf")

    plt.figure(7)
    plt.clf()
    plt.plot(f_data['loss'], linestyle="--", label="loss")
    plt.plot(f_data['loss'])
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(0.5, 2.5)
    plt.grid()
    plt.title("Testing Loss")
    plt.savefig(PLOT_PATH + alg_name + "Test_Loss.pdf")

    plt.figure(8)
    plt.clf()
    plt.plot(f_data['rec_loss'], linestyle="--", label="rec_loss")
    plt.plot(f_data['rec_loss'])
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(0.5, 2.5)
    plt.grid()
    plt.title("Training Client Reconstruction Loss")
    plt.savefig(PLOT_PATH + alg_name + "C_Rec_Loss_Training.pdf")
    #
    if RUNNING_ALG=="FedMEKT":
        plt.figure(9)
        plt.clf()
        plt.plot(f_data['kt_loss'])
        plt.plot(f_data['kt_loss'], linestyle="--", label="kt_loss")
        plt.legend()
        plt.xlabel("Global Rounds")
        plt.ylim(-1, 1.02)
        plt.grid()
        plt.title("Training Client Knowledge Transfer Loss")
        plt.savefig(PLOT_PATH + alg_name + "C_KT_Loss_Training.pdf")

        plt.figure(10)
        plt.clf()
        plt.plot(f_data['global_rec_loss'])
        plt.plot(f_data['global_rec_loss'], linestyle="--", label="root global_rec_loss")
        plt.legend()
        plt.xlabel("Global Rounds")
        plt.ylim(0.5, 2.5)
        plt.grid()
        plt.title("Training Global Reconstruction Loss")
        plt.savefig(PLOT_PATH + alg_name + "Global_Rec_Loss_Training.pdf")

        plt.figure(11)
        plt.clf()
        plt.plot(f_data['global_kt_loss'])
        plt.plot(f_data['global_kt_loss'], linestyle="--", label="global_kt_loss")
        plt.legend()
        plt.xlabel("Global Rounds")
        plt.ylim(-1, 1.02)
        plt.grid()
        plt.title("Training Global Knowledge Transfer Loss")
        plt.savefig(PLOT_PATH + alg_name + "Global_KT_Loss_Training.pdf")

    plt.figure(12)
    plt.clf()
    plt.plot(f_data['local_f1_acc'])
    plt.plot(f_data['local_f1_acc'], linestyle="--", label="local_f1_acc")
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(YLim, 1.02)
    plt.grid()
    plt.title("Testing Local F1 Accuracy")
    plt.savefig(PLOT_PATH + alg_name + "Testing_Local_F1_Accuracy.pdf")

    plt.figure(13)
    plt.clf()
    plt.plot(f_data['avg_local_f1_acc'])
    plt.plot(f_data['avg_local_f1_acc'], linestyle="--", label="avg_local_f1_acc")
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(YLim, 1.02)
    plt.grid()
    plt.title("Avg Testing Local F1 Accuracy")
    plt.savefig(PLOT_PATH + alg_name + "Avg_Testing_Local_F1_Accuracy.pdf")
    plt.show()

    # print("** Summary Results: ---- Training ----")
    # print("AVG Clients Specialization - Training:", f_data['cs_avg_data_train'])
    # print("AVG Clients Generalization - Training::", f_data['cg_avg_data_train'])
    # print("Root performance - Training:", f_data['root_train'])
    # print("** Summary Results: ---- Testing ----")
    # print("AVG Clients Specialization - Testing:", f_data['cs_avg_data_test'])
    # print("AVG Clients Generalization - Testing:", f_data['cg_avg_data_test'])
    # print("Root performance - Testing:", f_data['root_test'])

if __name__=='__main__':
    #
    # PLOT_PATH = "."+PLOT_PATH
    # RS_PATH = "."+RS_PATH
    rp = "."+complex_file_path
    # plot_from_file()
    print(rp)
    time_data = read_data(rp)
    data = time_data['time_complex']
    print(data)
    print('mean =', np.mean(data), ' median: = ', np.median(data))
