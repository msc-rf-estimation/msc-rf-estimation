# Import packages
import math
import numpy as np
c = 3e8

def get_distance(point1, point2):
    """
    Returns the Euclidean distance between two points.
    Each point should be a tuple or list of coordinates.
    """
    return math.dist(point1, point2)


def free_space_path_loss(d, freq):
    """Free-space path loss in dB.
    
    Args:
        d: distance in metres (scalar or array)
        freq: frequency in Hz (scalar)

        Uses speed of light c= 3e8 m/s defined at module level.

    Returns:
        Path loss in dB.
    """
    pl = 20 * np.log10(4 * np.pi * (d / (c/freq))) 
    return pl


def log_distance_path_loss(d, d0, n, pl_d0):
    """Log-distance path loss formula.

    Args:
        d: target distance(s), in metres (scalar or NumPy array). 
        d0: reference distance, in metres.
        n: path loss exponent, typical values 2.0-5.0; free space = 2, suburban = 2.5-3.5, dense urban = 3.5-5.0. 
        pl_d0: reference loss in dB. 

    Returns:
        Path loss in dB.
    """
    d = np.asarray(d)
    
    pl = pl_d0 + 10 * n * np.log10(d / d0)
    return pl
