import numpy as np  
from contested_rf.propagation.path_loss import log_distance_path_loss, free_space_path_loss  
from dataclasses import dataclass

@dataclass(frozen=True)
class JammerHypothesis:
    """Parameters describing a single static omnidirectional jammer."""
    x_j: float
    y_j: float
    p_tx_dbm: float

def likelihood(z, jammer_params, sensor_location, sigma):
    """Gaussian likelihood function. 
    
    Args:
        z: received power, dBm.
        jammer_params: dataclass.
        sensor_location: tuple of two floats in m. 
        sigma: shadow fading standard deviation (dB), default 4.0.
        
    Returns:
        Unnormalised Gaussian likelihood proportional form, dimensionless.
    """
    # Learner-model constants.
    d0 = 1.0        # Reference distance
    n = 2.0         # Learner's path loss exponent
    freq = 2.4e9    # Operating frequency 
    x_sensor, y_sensor = sensor_location

    # Auxilary calcs
    distance = np.sqrt((jammer_params.x_j - x_sensor)**2 + (jammer_params.y_j - y_sensor)**2)
    path_loss = log_distance_path_loss(distance, d0, n, free_space_path_loss(d0, freq))
    z_predicted = jammer_params.p_tx_dbm - path_loss
    r = z - z_predicted
    return np.exp(-r**2 / (2 * sigma **2))


