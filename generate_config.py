import sys
import random
random.seed(42)
import yaml
from attacks.attack_names import AttackNames
from detections.detection_names import DetectionNames
from dataset_names.dataset_names import DatasetNames

#################################################################
### Use for generating settings (e.g. via a bash script)

# Arguments:
#   1. DATASET
#   2. NUM_CLIENTS
#   3. PERCENT_MALICIOUS
#   4. ATTACK_METHOD
#   5. DETECTION_METHOD
#################################################################

# --- --- ---   --- --- --- --- ---   --- --- ---
# --- --- ---   Experiment Settings   --- --- ---
# --- --- ---   --- --- --- --- ---   --- --- ---

# Dataset (mnist or cifar10)
DATASET = DatasetNames(sys.argv[1])

# Number of clients
NUM_CLIENTS = int(sys.argv[2])

# Number of federation rounds
NUM_ROUNDS = 20 if DATASET == DatasetNames.mnist or (DATASET == DatasetNames.cifar10 and NUM_CLIENTS == 10) else 40

# Percentage of malicious clients
PERCENT_MALICIOUS = float(sys.argv[3])
print(f"PERCENT_MALICIOUS: {PERCENT_MALICIOUS}")

# Fraction of the clients to choose for training in one round. We always use 1.0
FRACTION_FIT = 1.0

# If true, plots about the loss and accuracy will be generated on wandb.
USE_WANDB = False

# Attack method to use
ATTACK_METHOD = AttackNames(sys.argv[4])

# Detection method to use
DETECTION_METHOD = DetectionNames(sys.argv[5])

# --- --- ---   --- --- --- --- ---   --- --- ---
# --- --- ---   --- --- --- --- ---   --- --- ---
# --- --- ---   --- --- --- --- ---   --- --- ---

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========
# This section contains settings and hyperparameters needed for the specific detections and attacks.
# ´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´
# ´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´
ADDITIONAL_DETECTION_CONFIG = {}
# ________________________________________________________________________________________________________________________________________________
if DETECTION_METHOD.value == DetectionNames.dagmm_detection.value:
    ADDITIONAL_DETECTION_CONFIG["do_data_collection"] = False   # Set to True to collect training data for the DAGMM model.
    ADDITIONAL_DETECTION_CONFIG["dagmm_output_dir"] = "./dagmm/dagmm/dagmm_train_data/" + DATASET.value + "/run_test_2"  # Output directory of the training data of the current run.

    ADDITIONAL_DETECTION_CONFIG["dagmm_threshold_path"] = "./dagmm/dagmm/dagmm_anomaly_threshold.yaml"
    ADDITIONAL_DETECTION_CONFIG["dagmm_ignore_up_to"] = 0   # Does not perform the detection in the first x rounds
    ADDITIONAL_DETECTION_CONFIG["dagmm_model_path"] = "./dagmm/dagmm/dagmm_model_mnist.pt"
    ADDITIONAL_DETECTION_CONFIG["dagmm_hyperparameters_path"] = "./dagmm/dagmm/dagmm_hyperparameters.yaml"
    ADDITIONAL_DETECTION_CONFIG["gmm_parameters_paths"] = {
         "cov": "./dagmm/dagmm/gmm_param_cov.pt",
         "mean": "./dagmm/dagmm/gmm_param_mean.pt",
         "mixture": "./dagmm/dagmm/gmm_param_mixture.pt",
    }
# ________________________________________________________________________________________________________________________________________________
elif DETECTION_METHOD.value == DetectionNames.std_dagmm_detection.value:
    ADDITIONAL_DETECTION_CONFIG["do_data_collection"] = False   # Use DAGMM for performing data collection
    ADDITIONAL_DETECTION_CONFIG["dagmm_output_dir"] = "-----" 
    ADDITIONAL_DETECTION_CONFIG["dagmm_threshold_path"] = "./dagmm/std_dagmm/dagmm_anomaly_threshold.yaml"
    ADDITIONAL_DETECTION_CONFIG["dagmm_ignore_up_to"] = 0   # Does not perform the detection in the first x rounds
    ADDITIONAL_DETECTION_CONFIG["dagmm_model_path"] = "./dagmm/std_dagmm/dagmm_model_mnist.pt"
    ADDITIONAL_DETECTION_CONFIG["dagmm_hyperparameters_path"] = "./dagmm/std_dagmm/dagmm_hyperparameters.yaml"
    ADDITIONAL_DETECTION_CONFIG["gmm_parameters_paths"] = {
         "cov": "./dagmm/std_dagmm/gmm_param_cov.pt",
         "mean": "./dagmm/std_dagmm/gmm_param_mean.pt",
         "mixture": "./dagmm/std_dagmm/gmm_param_mixture.pt",
    }
# ________________________________________________________________________________________________________________________________________________
elif DETECTION_METHOD.value == DetectionNames.delta_dagmm_detection.value:
    ADDITIONAL_DETECTION_CONFIG["do_data_collection"] = False   # Set to True to collect training data for the Delta-DAGMM model.
    ADDITIONAL_DETECTION_CONFIG["do_test_data_collection"] = False   # Set to True to collect test data (additionally collects the global model for each iteration)
    ADDITIONAL_DETECTION_CONFIG["dagmm_output_dir"] = "./dagmm/delta_dagmm/dagmm_train_data/" + DATASET.value + "/" + str(NUM_CLIENTS) + "/run_test_2"  # Output directory of the training data of the current run.

    model_path = "./dagmm/delta_dagmm/models/" + DATASET.value + "/" + str(NUM_CLIENTS)
    ADDITIONAL_DETECTION_CONFIG["dagmm_threshold_path"] = model_path + "/dagmm_anomaly_threshold.yaml"
    ADDITIONAL_DETECTION_CONFIG["dagmm_ignore_up_to"] = 2   # Does not perform the detection in the first x round(s). This is important for Delta-DAGMM since especially in the first iteration, the global model is randomly initialized by the server. Thus it is good to skip it.
    ADDITIONAL_DETECTION_CONFIG["dagmm_model_path"] =  model_path + "/model.pt"
    ADDITIONAL_DETECTION_CONFIG["dagmm_hyperparameters_path"] = model_path + "/dagmm_hyperparameters.yaml"
    ADDITIONAL_DETECTION_CONFIG["gmm_parameters_paths"] = {
         "cov": model_path + "/gmm_param_cov.pt",
         "mean": model_path + "/gmm_param_mean.pt",
         "mixture": model_path + "/gmm_param_mixture.pt",
    }
# ________________________________________________________________________________________________________________________________________________
elif DETECTION_METHOD.value == DetectionNames.rffl_detection.value:
    ADDITIONAL_DETECTION_CONFIG["alpha"] = 0.95
    ADDITIONAL_DETECTION_CONFIG["beta"] = 1/(3*NUM_CLIENTS)
    ADDITIONAL_DETECTION_CONFIG["gamma"] = 0.5 if DATASET.value == "mnist" else 0.15  # TODO see if these values are good (0.5 for MNIST and 0.15 for CIFAR10 specified by the authors)
# ________________________________________________________________________________________________________________________________________________
elif DETECTION_METHOD.value == DetectionNames.fgfl_detection.value:
    ADDITIONAL_DETECTION_CONFIG["stds"] = 2.1    # The amount of standard deviations to substract from the mean. Higher -> higher precision, lower recall. Lower -> lower precision, higher recall
# ________________________________________________________________________________________________________________________________________________
elif DETECTION_METHOD.value == DetectionNames.fdfl_detection.value:
    ADDITIONAL_DETECTION_CONFIG["tau"] = 0.4    # If the cosine similarity of two submitted data distributions for two clients of the same cluster is smaller than tau,
                                                # i.e. they are not very similar, the flag counter will be increased (if flag is very high, it will be marked as a free rider).
    ADDITIONAL_DETECTION_CONFIG["n_clusters"] = 5 if NUM_CLIENTS == 100 else 3       # Use 5 for 100 clients and 3 for 10 clients
    method = "weak"     # Perform either 'weak' or 'strong' imitation of label distributions.
    with open("./config/fake_label_distribution.yaml", "w") as f:
        yaml.dump({"method": method}, f)
# ________________________________________________________________________________________________________________________________________________
elif DETECTION_METHOD.value == DetectionNames.viceroy_detection.value:
    ADDITIONAL_DETECTION_CONFIG["omega"] = 0.525    # History decay factor (value taken from the paper)
    ADDITIONAL_DETECTION_CONFIG["eta"] = 0.2        # Reputation update factor (value taken from the paper)
    ADDITIONAL_DETECTION_CONFIG["kappa"] = 0.5      # Confidence parameter of FoolsGold.
    ADDITIONAL_DETECTION_CONFIG["skip_first_round"] = True  # Specify if the first round should be skipped. 
                                                            # This is a good practice since the initial global model is randomly initialized.
                                                            # Thus, the gradient calculation can be messy.
    ADDITIONAL_DETECTION_CONFIG["free_rider_threshold"] = 0.1   # Specify the threshold that separates benign clients from free riders.
                                                                # The values it is compared to range from 0 to 1, where 0 denotes a very suspicious client
                                                                # and 1 a very unsuspicious one.
# ________________________________________________________________________________________________________________________________________________
elif DETECTION_METHOD.value == DetectionNames.wef_detection.value:
    ADDITIONAL_DETECTION_CONFIG["epsilon"] = 0.02       # Determines how strict the threshold distinguishes between benign clients and free riders.
                                                        # Choosing a low value results in low FP but may miss free riders whereas a high value may results in FPs.
# ________________________________________________________________________________________________________________________________________________
print("Using the detection method: ", DETECTION_METHOD.value)
if "dagmm" in DETECTION_METHOD.value and ADDITIONAL_DETECTION_CONFIG and ADDITIONAL_DETECTION_CONFIG["do_data_collection"]:
    if PERCENT_MALICIOUS != 0.0 and ATTACK_METHOD != AttackNames.no_attack:
        print("\n\n####### Warning: Performing data collection with malicious clients! Only use if intended! #######\n\n")
# ´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´
# ´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ADDITIONAL_ATTACK_CONFIG = {}
if ATTACK_METHOD.value == "random_weights":
    ADDITIONAL_ATTACK_CONFIG["R"] = 1e-2        # 1e-2 is best against Delta-DAGMM

if ATTACK_METHOD.value == "advanced_free_rider":
    ADDITIONAL_ATTACK_CONFIG["n"] = NUM_CLIENTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# Store all the necessary settings
from util import store_necessary_settings
store_necessary_settings(NUM_CLIENTS, PERCENT_MALICIOUS, DATASET, ATTACK_METHOD, ADDITIONAL_ATTACK_CONFIG, DETECTION_METHOD, ADDITIONAL_DETECTION_CONFIG, NUM_ROUNDS, FRACTION_FIT, USE_WANDB)