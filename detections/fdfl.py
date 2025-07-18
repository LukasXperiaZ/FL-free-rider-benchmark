from detections.detection import Detection
from flwr.common import parameters_to_ndarrays
import numpy as np
import json

class FDFLDetection(Detection):
    def __init__(self, config):
        self.config = config

    def detect(self, server_round, client_ids, client_updates, client_metrics, global_model):
        # 0. Parse the counts of labels of all clients
        label_counts = [] 
        for client_metric in client_metrics:
            label_counts_dict = json.loads(client_metric.get("label_counts"))
            label_counts.append(label_counts_dict)

        # 1. Perform K-means clustering on the weights of the client models

        # 2. For each cluster, check for each client if its submitted data distribution is similar or different to other clients.

        # TODO
