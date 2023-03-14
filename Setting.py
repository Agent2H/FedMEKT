"FIX PROGRAM SETTINGS"
import random
import numpy as np
import torch

READ_DATASET = True  # True or False => Set False to generate dataset.

Subset = True# True: Using fraction of users
One_Layer = False
Global_Classifier= False

PLOT_PATH = "./figs/"
RS_PATH = "./results/"
FIG_PATH = "./results_fig/"
# FIG_PATH = "./results_KSC2021/"
PLOT_PATH_FIG="./figs/"
# PLOT_PATH_FIG="./KSC_Figs/"
data_path = "./data/"
#Dataset selection
DATASETS= ["mhealth","opp","ur_fall"]
DATASET = DATASETS[2]
#Algorithm selection
RUNNING_ALGS = ["mmFedAvg","mmFedProx","mmMOON","FedMEKT-S","FedMEKT-C"]

RUNNING_ALG = RUNNING_ALGS[4]

#Model selection
models=["split_LSTM"]
MODEL_AE=models[0]
# #Metric selection
# CDKT_metrics = ["KL","Norm2","JSD","Cos","Con"]
# Global_CDKT_metric = CDKT_metrics[0]   # Global distance metric
# Local_CDKT_metric = CDKT_metrics[0]    # Local distance metric

PUBLIC_RATIO=1
#Algorithm Parameter
Num_neurons=32
ALPHA = 100
alpha =0.1#trade-of parameter of local training loss
beta= 0.04#trade-off parameter of global distillation loss (Mnist:rep+full loss 0.2)
eta = 0.1
gamma=0.04
mu=0.001#contrastive hyperparameter
muy=0.001# mmFedProx hyperparameter
lamda = 1
sigma= 0.02

train_ratio = 0.11
train_supervised_ratio =0.11
DCCAE_lamda = 0.01

local_learning_rate = 0.004
# local_classifier_learning_rate=0.001
global_learning_rate = 0.006
global_ae_learning_rate = 0.002
global_ae_distill_epoch= 2
global_generalized_epochs = 5
# local_classifier_epochs=2
LOCAL_EPOCH = 2
NUM_GLOBAL_ITERS = 100# Number of global rounds

#Eval window sequence for evaluation
EVAL_WIN = 2000
#Clientnum
NUM_CLIENT_A= 10
NUM_CLIENT_B= 10
NUM_CLIENT_AB= 10

#Modality Setting
test_modalities=["A","B"]
test_modality=test_modalities[0]
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
    MODALITY_B = modality_urfall[2]

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
 Frac_users = 0.33 #20% of users will be selected
 # Frac_users = 0.1  #20% of users will be selected
else:
 N_clients = NUM_CLIENT_A+NUM_CLIENT_B+NUM_CLIENT_AB
 Frac_users = 1.  #All users will be selected
K_Levels =1







rs_file_path = "{}_{}_L{}_T{}_A{}_B{}_AB{}_I{}_Ma{}_Mb{}_alpha{}_eta{}_beta{}_gamma{}_SS{}_ratio{}_depoch{}_lepoch{}_onelayer{}_globalclsDr{}_publicratio{}.h5".format(RUNNING_ALG, DATASET,label_modality,test_modality,NUM_CLIENT_A,NUM_CLIENT_B,NUM_CLIENT_AB,NUM_GLOBAL_ITERS,
                                MODALITY_A,MODALITY_B,alpha,eta,beta,gamma,Subset,train_ratio,global_ae_distill_epoch,LOCAL_EPOCH,One_Layer,Global_Classifier,PUBLIC_RATIO )
rs_file_path = FIG_PATH + rs_file_path
PLOT_PATH += DATASET+'_'
print("Result Path ", rs_file_path)

# complex_file_path = "{}_{}_I{}_time_.h5".format(DATASET, RUNNING_ALG, NUM_GLOBAL_ITERS)





