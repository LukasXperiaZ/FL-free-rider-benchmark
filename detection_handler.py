from detections.detection_names import DetectionNames
from detections.detection import Detection

class DetectionHandler:
    def __init__(self, method, config):
        self.method = method.lower()
        self.config = config
        if self.method == DetectionNames.sample_detection.value:
            self.detector = SampleDetection(config)
        elif self.method == DetectionNames.dagmm_detection.value:
            from detections.dagmm_detection import DAGMMDetection
            self.detector = DAGMMDetection(config)
        elif self.method == DetectionNames.no_detection.value:
            print("### Using no detection ###")
            self.method = None
            self.detector = None
        else:
            raise NameError(f"Detection {self.method} not known!")
        # Add more methods here

    def detect_anomalies(self, server_round, client_ids, client_updates):
        if self.method is None:
            # no filtering
            return client_ids
        # Filter client updates
        return self.detector.detect(server_round, client_ids, client_updates)

class SampleDetection(Detection):
    """
    Sample Detection illustrating the main components of a detection.
    """
    def __init__(self, config):
        # initialize the autoencoder, estimator network, etc.
        self.config = config

    def detect(self, server_round, client_ids, client_updates):
        # Sample detection, always removes the last client
        return client_ids[:-1]