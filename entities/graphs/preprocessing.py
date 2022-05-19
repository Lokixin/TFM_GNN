import numpy as np
from sklearn import preprocessing


def normalize(data: np.ndarray, dim=1, norm="l2") -> np.ndarray:
    """Normalizes the data channel by channel

    Args:
        data (np.ndarray): Raw EEG signal
        dim (int, optional): 0 columns, 1 rows. The dimension where 
        the mean and the std would be computed.

    Returns:
        np.ndarray: Channel-wise normalized matrix
    """
    normalized_data = preprocessing.normalize(data, axis=dim, norm=norm)
    return normalized_data
    
    