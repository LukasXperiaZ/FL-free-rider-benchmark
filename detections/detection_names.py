from enum import Enum

class DetectionNames(Enum):
    no_detection = "no_detection"
    sample_detection = "sample"
    dagmm_detection = "dagmm"
    std_dagmm_detection = "std_dagmm"
    delta_dagmm_detection = "delta_dagmm"