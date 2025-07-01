from enum import Enum

class AttackNames(Enum):
    random_weights_attack = "random_weights"
    advanced_delta_weights_attack = "advanced_delta_weights"
    advanced_free_rider_attack = "advanced_free_rider"