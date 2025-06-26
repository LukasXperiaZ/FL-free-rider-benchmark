from task import set_weights
from client_app import BenignClient
import torch

class RandomWeightsAttackClient(BenignClient):

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        # Random Weights attack
        R = self.config.get("R")
        new_params = []
        for p in self.net.parameters():
            shape = p.data.shape
            rand_tensor = torch.empty(shape).uniform_(-R,R)
            new_params.append(rand_tensor.numpy())

        return (new_params, 
                len(self.trainloader),
                {"partition_id": self.partition_id}
            )