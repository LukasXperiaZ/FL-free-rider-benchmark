from detections.detection_names import DetectionNames
from detections.detection import Detection, DetectionAfterAggregation

class DetectionHandler:
    def __init__(self, method, config):
        self.method = method.lower()
        self.config = config
        if self.method == DetectionNames.sample_detection.value:
            self.detector = SampleDetection(config)
        elif self.method == DetectionNames.dagmm_detection.value:
            from detections.dagmm_detection import DAGMMDetection
            self.detector = DAGMMDetection(config)
        elif self.method == DetectionNames.std_dagmm_detection.value:
            from detections.std_dagmm_detection import STDDAGMMDetection
            self.detector = STDDAGMMDetection(config)
        elif self.method == DetectionNames.delta_dagmm_detection.value:
            from detections.delta_dagmm_detection import DeltaDAGMMDetection
            self.detector = DeltaDAGMMDetection(config)
        elif self.method == DetectionNames.fgfl_detection.value:
            from detections.fgfl import FGFLDetection
            self.detector = FGFLDetection(config)
        elif self.method == DetectionNames.fdfl_detection.value:
            from detections.fdfl import FDFLDetection
            self.detector = FDFLDetection(config)
        elif self.method == DetectionNames.rffl_detection.value:
            # RFFL handles the detection itself
            self.detector = None
        elif self.method == DetectionNames.no_detection.value:
            print("### Using no detection ###")
            self.method = None
            self.detector = None
        else:
            raise NameError(f"Detection {self.method} not known!")

        # Check whether the detection should be performed after the aggregation step
        if isinstance(self.detector, DetectionAfterAggregation):
            self.after_aggregation = True
        else:
            self.after_aggregation = False

    def set_aggregated_global_model(self, aggregated_global_model):
        # In case the detection is a Detection that is performed after aggregation
        if not isinstance(self.detector, DetectionAfterAggregation):
            raise RuntimeError(f"The detection {self.method} is not a DetectionAfterAggregation!")
        
        self.detector.set_aggregated_global_model(aggregated_global_model)

    def detect_anomalies(self, server_round, client_ids, client_updates, client_metrics, global_model):
        if self.method is None:
            # no filtering
            return client_ids
        # Filter client updates
        return self.detector.detect(server_round, client_ids, client_updates, client_metrics, global_model)

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