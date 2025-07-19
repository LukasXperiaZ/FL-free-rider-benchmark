from collections import Counter
import numpy as np
import yaml

with open("./config/fake_label_distribution.yaml", "r") as f:
    fake_label_dist_config = yaml.safe_load(f)
method = fake_label_dist_config["method"]

def fake_label_distribution(trainloader):
    # Note that we use the trainloader of a free rider attacker, altough in reality, it does not have training data.

    if method == "weak":
        return _weak_faking(trainloader)
    elif method == "strong":
        return _strong_faking(trainloader)
    else:
        raise RuntimeError("Method for fake_label_distribution must be either 'strong' or 'weak', not: ", method)
    

def _weak_faking(trainloader):
    # We ONLY use the trainloader to get rough estimates on which class labels are there.

    # Assume that an attacker knows (some) class labels
    class_counts = Counter()

    for _, label in enumerate(trainloader):
        label_tensor = label['label']
        class_counts.update(label_tensor.tolist())

    counts_dict = dict(class_counts)

    # Assume that an attacker makes just an educated guess about mean and std.
    mean = 50
    std = 10

    # Estimate the amount of class lables using the mean and std.
    est_counts_dict = {}
    est_amounts_raw = np.random.normal(loc=mean, scale=std, size=len(counts_dict.keys()))
    # Process the raw estimated amounts.
    est_amounts = np.maximum(0, np.round(est_amounts_raw)).astype(int)
    for i, key in enumerate(counts_dict.keys()):
        est_counts_dict[key] = int(est_amounts[i])

    return est_counts_dict

def _strong_faking(trainloader):
    # We ONLY use the trainloader to get rough estimates on which class labels are there and how many samples do clients have roughly.

    # Assume that an attacker knows (some) class labels
    class_counts = Counter()

    for _, label in enumerate(trainloader):
        label_tensor = label['label']
        class_counts.update(label_tensor.tolist())

    counts_dict = dict(class_counts)

    # Assume that an attacker can estimate the mean and std of the amounts of samples other clients may have per class.
    np_array = np.array(list(counts_dict.values()))
    mean = np.mean(np_array)
    std = np.std(np_array, ddof=1)

    # Estimate the amount of class lables using the mean and std.
    est_counts_dict = {}
    est_amounts_raw = np.random.normal(loc=mean, scale=std, size=len(counts_dict.keys()))
    # Process the raw estimated amounts.
    est_amounts = np.maximum(0, np.round(est_amounts_raw)).astype(int)
    for i, key in enumerate(counts_dict.keys()):
        est_counts_dict[key] = int(est_amounts[i])

    return est_counts_dict
