from detections.detection import Detection
from flwr.common import parameters_to_ndarrays
import numpy as np
import pickle

class WEFDetection(Detection):
    def __init__(self, config):
        self.config = config
        # A dict mapping from client ids to their WEF matrices
        self.wef_matrix = {}
        # A dict mapping from client ids to their previous penultimate layer (used for calculating how weights change)
        self.w_t = {}
        self.epsilon = config.get("epsilon")

    def detect(self, server_round, client_ids, client_updates, client_metrics, global_model):
        global_model_layers = parameters_to_ndarrays(global_model)

        # 0. Initialize the WEF-matrix and the first weights submitted by the clients (w_t)
        if not self.wef_matrix:
            example_matrix = self._model_to_penultimate_layer(global_model_layers)
            for cid, model in zip(client_ids, client_updates):
                self.wef_matrix[cid] = np.zeros_like(example_matrix)

                w_t_client = self._model_to_penultimate_layer(model)
                self.w_t[cid] = w_t_client

            # Do not perform the detection in the first round, as no change in the model weights of clients can be computed
            return client_ids 
        
        # 1. Determine the thresholds (corresponds to a_i^{t'+1} in the paper)
        thresholds = {}
        variations = {}
        for cid, model in zip(client_ids, client_updates):
            w_t_1 = self._model_to_penultimate_layer(model)
            var_client = np.abs(self.w_t[cid] - w_t_1)
            variations[cid] = var_client
            thresholds[cid] = np.average(var_client)

            # Update w_t for the next round.
            self.w_t[cid] = w_t_1

        # 2. Collect the WEF-matrix
        for cid in client_ids:
            vars_client = variations[cid]
            threshold_client = thresholds[cid]
            WEF_matrix_client = self.wef_matrix[cid]

            assert len(vars_client) == len(WEF_matrix_client)

            for i, var in enumerate(vars_client):
                if var > threshold_client:
                    # Increment the WEF matrix entry
                    WEF_matrix_client[i] += 1

        # 3. Separate clients (O(n^2), n=|clients|)
        Dev = {}
        dis = {}
        cos = {}
        avg = {}
        for i in client_ids:
            wef_i = self.wef_matrix[i]
            # Sum of all euclidean distances
            dis_i = 0
            # Sum of all cosine distances
            cos_i = 0
            for j in client_ids:
                if i != j:
                    wef_j = self.wef_matrix[j]

                    # 3.1 Calculate the Euclidean distance
                    euclidean = np.linalg.norm(wef_i - wef_j)
                    dis_i += euclidean
                    
                    # 3.2 Calculate the cosine similarity
                    cosine = np.dot(wef_i, wef_j) / (np.linalg.norm(wef_i)*np.linalg.norm(wef_j))
                    cos_i += cosine

            dis[i] = dis_i
            cos[i] = cos_i

            # Calculate the average frequency of wef_i
            avg_i = np.average(wef_i)
            avg[i] = avg_i

        
        # Calculate the deviations
        dis_avg = np.average(list(dis.values()))
        cos_avg = np.average(list(cos.values()))
        avg_avg = np.average(list(avg.values()))

        dis_devs = {}
        cos_devs = {}
        avg_devs = {}
        #print(f"Deviations of i:\tdis_dev,\tcos_dev,\tavg_dev")
        for i in client_ids:
            dis_dev = np.abs(dis_avg - dis[i])
            cos_dev = np.abs(cos_avg - cos[i])
            avg_dev = np.abs(avg_avg - avg[i])

            dis_devs[i] = dis_dev
            cos_devs[i] = cos_dev
            avg_devs[i] = avg_dev

            #print(f"Deviations of {i}:\t{dis_dev:.6f},\t{cos_dev:.6f},\t{avg_dev:.6f}")

        # Normalize dis_i, cos_i and dev_i and add them to obtain Dev_i
        dis_dev_sum = np.sum(list(dis_devs.values()))
        cos_dev_sum = np.sum(list(cos_devs.values()))
        avg_dev_sum = np.sum(list(avg_devs.values()))
        for i in client_ids:
            Dev[i] = dis_devs[i]/dis_dev_sum + cos_devs[i]/cos_dev_sum + avg_devs[i]/avg_dev_sum

        #print("\nDeviations:\n", Dev, "\n")
        # 3.1 Calculate the reputation threshold
        xi = np.max(list(Dev.values())) - self.epsilon
        #print("xi: ", xi)

        # 3.2 Filter clients
        benign_clients = [cid for cid in client_ids if Dev[cid] < xi]

        return benign_clients

    def _model_to_penultimate_layer(self, model):
        last_layer_weights = model[-2]
        last_layer_bias = model[-1]
        penultimate_layer = np.concatenate([last_layer_weights.flatten(), last_layer_bias.flatten()])
        return penultimate_layer