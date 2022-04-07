"FIX PROGRAM SETTINGS"
import random
import numpy as np
import torch

READ_DATASET = True  # True or False => Set False to generate dataset.

Subset = True# True: Using fraction of users


PLOT_PATH = "./figs/"
RS_PATH = "./results/"
FIG_PATH = "./results_fig/"
# FIG_PATH = "./results_KSC2021/"
PLOT_PATH_FIG="./figs/"
# PLOT_PATH_FIG="./KSC_Figs/"
data_path = "./data/"
#Dataset selection
DATASETS= ["mhealth","opp","ur_fall"]
DATASET = DATASETS[1]
#Algorithm selection
RUNNING_ALGS = ["mmFedAvg","mmFedEKT"]
RUNNING_ALG = RUNNING_ALGS[1]

#Model selection
models=["split_LSTM","DCCAE_LSTM"]
MODEL_AE=models[1]
#Metric selection
CDKT_metrics = ["KL","Norm2","JSD"]
Global_CDKT_metric = CDKT_metrics[0]   # Global distance metric
Local_CDKT_metric = CDKT_metrics[1]    # Local distance metric

#Algorithm Parameter
ALPHA = 100
alpha =0.3# trade-off parameter of local training loss
beta =0.3# trade-off parameter of global distillation loss (Mnist:rep+full loss 0.2)
gamma=10
eta = 10
lamda = 0.1
train_ratio = 0.11
train_supervised_ratio =0.11
DCCAE_lamda = 0.01

local_learning_rate = 0.001
global_learning_rate = 0.001
global_ae_learning_rate = 0.001
global_ae_distill_epoch= 2
global_generalized_epochs = 5
LOCAL_EPOCH = 2
NUM_GLOBAL_ITERS = 100 # Number of global rounds

#Eval window sequence for evaluation
EVAL_WIN = 2000
#Clientnum
NUM_CLIENT_A= 0
NUM_CLIENT_B= 0
NUM_CLIENT_AB= 30

#Modality Setting
test_modalities=["A","B"]
test_modality=test_modalities[1]
label_modalities = ["A","B","AB"]
label_modality = label_modalities[2]
modality_opp=["acce","gyro"]
modality_mheath=["acce","gyro","mage"]
modality_urfall=["acce","rgb","depth"]
if DATASET == "opp":
    MODALITY_A = modality_opp[0]
    MODALITY_B = modality_opp[1]
elif DATASET == "mhealth":
    MODALITY_A = modality_mheath[0]
    MODALITY_B = modality_mheath[1]
elif DATASET == "ur_fall":
    MODALITY_A = modality_urfall[0]
    MODALITY_B = modality_urfall[1]

#Rep size setting
rep_size = 0
if DATASET == "opp":
    rep_size = 10
elif DATASET == "mhealth":
    rep_size = 4
elif DATASET == "ur_fall":
    if "acce" in MODALITY_A or "acce" in MODALITY_B:
        rep_size = 2
    else:
        rep_size = 4

SEED = 1
random.seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
### Agorithm Parameters ###
if Subset:
 N_clients = NUM_CLIENT_A+NUM_CLIENT_B+NUM_CLIENT_AB
 Frac_users = 0.1  #20% of users will be selected
else:
 N_clients = NUM_CLIENT_A+NUM_CLIENT_B+NUM_CLIENT_AB
 Frac_users = 1.  #All users will be selected
K_Levels =1







rs_file_path = "{}_{}_I{}_a{}_b{}_SS{}_gm{}_lm{}.h5".format(RUNNING_ALG, DATASET, NUM_GLOBAL_ITERS,
                                alpha,beta,Subset,Global_CDKT_metric,Local_CDKT_metric )
rs_file_path = FIG_PATH + rs_file_path
PLOT_PATH += DATASET+'_'
print("Result Path ", rs_file_path)

# complex_file_path = "{}_{}_I{}_time_.h5".format(DATASET, RUNNING_ALG, NUM_GLOBAL_ITERS)





