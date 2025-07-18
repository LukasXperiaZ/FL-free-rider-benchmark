from task import set_weights
from attacks.attack_client import AttackClient
import torch
import json

class RandomWeightsAttackClient(AttackClient):

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        # Random Weights attack
        R = self.config.get("R")
        new_params = []
        for p in self.net.parameters():
            shape = p.data.shape
            rand_tensor = torch.empty(shape).uniform_(-R,R)
            new_params.append(rand_tensor.numpy())

        label_counts_dict = self._get_label_distribution(self.trainloader)
        return self._fit_return(new_params, len(self.trainloader.dataset), 0.5, self.partition_id, json.dumps(label_counts_dict))
