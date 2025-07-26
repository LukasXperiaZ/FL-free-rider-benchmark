"""pytorch-example: A Flower / PyTorch app."""

import torch
from task import Net, get_weights, load_data, set_weights, test, train
from attacks.attack_names import AttackNames

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, RecordDict
from collections import Counter

import yaml
import json

# Load malicious client IDs from file
with open("./config/malicious_clients.yaml", "r") as f:
    malicious_config = yaml.safe_load(f)
malicious_ids = malicious_config.get("malicious_clients", [])


# Define Flower Client and client_fn
class BenignClient(NumPyClient):
    """A simple client that showcases how to use the state.

    It implements a basic version of `personalization` by which
    the classification layer of the CNN is stored locally and used
    and updated during `fit()` and used during `evaluate()`.
    """

    def __init__(
        self, net, client_state: RecordDict, trainloader, valloader, local_epochs, partition_id, config
    ):
        self.net: Net = net
        self.client_state = client_state
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.local_layer_name = "classification-head"
        self.partition_id = partition_id
        self.config = config

    def fit(self, parameters, config):
        """Train model locally.

        The client stores in its context the parameters of the last layer in the model
        (i.e. the classification head). The classifier is saved at the end of the
        training and used the next time this client participates.
        """

        # Apply weights from global models (the whole model is replaced)
        set_weights(self.net, parameters)

        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            #lr=float(config["lr"]),
            device=self.device,
        )

        label_counts_dict = self._get_label_distribution(self.trainloader)
        # Return locally-trained model and metrics
        return self._fit_return(get_weights(self.net), len(self.trainloader.dataset), train_loss, self.partition_id, json.dumps(label_counts_dict))
    
    def _fit_return(self, weights, n_samples, train_loss, partition_id, label_counts):
        return (
            weights,
            n_samples,
            {"train_loss": train_loss,
             "partition_id": partition_id,
             "label_counts": label_counts},
        )
    
    def _get_label_distribution(self, trainloader):
        """
        Computes the data distribution (counts of samples per class) from a trainloader.

        Args:
            trainloader (torch.utils.data.DataLoader):  The DataLoader for your training data.
                                                        Assumes that each batch yields (data, labels).

        Returns:
            counts_dict:    Return a dictionary mapping from label to amount
                            E.g.: {3: 6131, 7: 6265, 1: 6742, 6: 5918, 4: 5842, 2: 5958, 9: 5949, 0: 5923, 5: 5421, 8: 5851}
        """
        class_counts = Counter()

        for _, label in enumerate(trainloader):
            label_tensor = label['label']
            class_counts.update(label_tensor.tolist())

        counts_dict = dict(class_counts)

        return counts_dict


    """ def evaluate(self, parameters, config):
        Evaluate the global model on the local validation set.

        Note the classification head is replaced with the weights this client had the
        last time it trained the model.
        
        set_weights(self.net, parameters)
        # Override weights in classification layer with those this client
        # had at the end of the last fit() round it participated in
        self._load_layer_weights_from_state()
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy} """


# Load attack method from file
attack_config = None
with open("./config/attack_method.yaml", "r") as f:
    attack_config = yaml.safe_load(f)

def client_fn(context: Context):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    client_state = context.state

    # Return Client instance
    # We pass the state to persist information across
    # participation rounds. Note that each client always
    # receives the same Context instance (it's a 1:1 mapping)
    attack_method = attack_config.get("attack_method", [])
    if partition_id in malicious_ids:
        #print(f"Client {partition_id} is malicious")
        if attack_method == AttackNames.random_weights_attack.value:
            from attacks.random_weights_attack import RandomWeightsAttackClient
            return RandomWeightsAttackClient(net, client_state, trainloader, valloader, local_epochs, partition_id=partition_id, config=attack_config).to_client()
        elif attack_method == AttackNames.advanced_delta_weights_attack.value:
            from attacks.advanced_delta_weights_attack import AdvancedDeltaWeightsAttack
            return AdvancedDeltaWeightsAttack(net, client_state, trainloader, valloader, local_epochs, partition_id=partition_id, config=attack_config).to_client()
        elif attack_method == AttackNames.advanced_free_rider_attack.value:
            from attacks.advanced_free_rider_attack import AdvancedFreeRiderAttack
            return AdvancedFreeRiderAttack(net, client_state, trainloader, valloader, local_epochs, partition_id=partition_id, config=attack_config).to_client()
        elif attack_method == AttackNames.adaptive_attack.value:
            from attacks.adaptive_attack import AdaptiveAttack
            return AdaptiveAttack(net, client_state, trainloader, valloader, local_epochs, partition_id, attack_config).to_client()
        elif attack_method == AttackNames.no_attack.value:
            return BenignClient(net, client_state, trainloader, valloader, local_epochs, partition_id=partition_id, config=attack_config).to_client()
        else:
            raise RuntimeError(f"Attack method '{attack_method}' not known!")

    else:
        return BenignClient(net, client_state, trainloader, valloader, local_epochs, partition_id=partition_id, config=attack_config).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)