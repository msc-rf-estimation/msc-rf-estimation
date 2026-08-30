import numpy as np  

def sinr_from_powers(p_signal_dbm, p_interference_dbm, p_noise_dbm): 
    """Compute SINR in dB from pre-computed received powers in dBM
    
    Args:
        p_signal_dbm: received signal at sensor, after propagation loss
        p_interference_dbm: received interference at sensor, after propagaton loss
        p_noise_dbm: thermal noise floor (including receiver noise figure) in dBm
    
    Returns: 
        SINR in dB.
    """
    
    # Convert dBm to linear Milliwatts (mW)
    # Formula: P_mw = 10**(P_dbm / 10)
    p_interference_dbm = np.atleast_1d(p_interference_dbm)
    s_mw = 10**(p_signal_dbm / 10)
    i_mw = 10**(p_interference_dbm / 10)
    n_mw = 10**(p_noise_dbm / 10)

    # Calculate SINR in linear scale
    sinr_linear = s_mw / (i_mw + n_mw)

    # Convert linear SINR to dB
    # Formula: SINR_db = 10 * log10(SINR_linear)
    sinr_db = 10 * np.log10(sinr_linear)

    return sinr_db