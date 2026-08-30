"""Experimental parameters — single source of truth for the simulation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """All experimental parameters for the contested RF estimation simulation."""
    # Propagation
    frequency_hz: float = 2.4e9
    reference_distance_m: float = 1.0
    path_loss_exponent_truth: float = 2.5
    path_loss_exponent_learner: float = 2.0
    
    # Shadow fading
    shadow_fading_sigma_db: float = 4.0
    shadow_fading_decorrelation_m: float = 50.0
    
    # Operating area (large enough for meaningful spatial structure, small enough for 480 runs to complete quickly)
    area_extent_m: float = 2000.0
    grid_resolution_m: float = 20.0
    
    # Sensor / receiver
    bandwidth_hz: float = 20e6
    temperature_k: float = 290
    receiver_noise_figure_db: float = 6.0