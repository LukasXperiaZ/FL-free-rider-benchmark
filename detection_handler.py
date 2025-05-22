class DetectionHandler:
    def __init__(self, method, config):
        self.method = method.lower()
        self.config = config
        if self.method == 'sample':
            self.detector = SampleDetection(config)
        elif self.method == 'dagmm':
            from detections.dagmm_detection import DAGMMDetection
            self.detector = DAGMMDetection(config)
        elif self.method == 'none':
            self.detector = None
        # Add more methods here

    def detect_anomalies(self, server_round, client_ids, client_updates):
        if self.method == 'none':
            # no filtering
            return client_ids
        # Filter client updates
        return self.detector.detect(server_round, client_ids, client_updates)
    

class SampleDetection:
    def __init__(self, config):
        # initialize the autoencoder, estimator network, etc.
        pass

    def detect(self, server_round, client_ids, client_updates):
        # Sample detection, always removes the last client
        return client_ids[:-1]