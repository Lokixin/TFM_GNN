import torch
import numpy as np
from abc import ABC, abstractmethod

from scipy import stats
from scipy.signal import welch
from scipy.integrate import simps

from .preprocessing import normalize
from .constants import SF


class BaseNodeExtractor(ABC):
    
    @abstractmethod
    def __init__(self) -> None: ...
    
    @abstractmethod
    def extract_features(self, data: np.ndarray): ...
    
    
class RawExtractor(BaseNodeExtractor):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.normalize = kwargs.get("normalize", False)
        self.th = kwargs.get("th", None)
        
    def extract_features(self, data):
        if self.normalize:
            data = normalize(data)
            
        if self.th:
            data = np.where(
                abs(data) < self.th, 0, data
            )
            
        node_features = torch.from_numpy(data)
        return node_features
    
    
class StadisticalMomentsExtractor(BaseNodeExtractor):
    
    def __init__(self) -> None:
        """StadisticalMommentExtractor computes the: 
            - Mean
            - Std
            - Entropy
            - Variance
            - Skewness
            - Kurtosis
            
        For every channel independently and returns a tensor
        of NUMBER_CHANNELS x STATISTICAL_MOMENTS (19x6) 
        """
        super().__init__()
        
        
    def extract_features(self, data):
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        variance = np.var(data, axis=1)
        entropy = stats.differential_entropy(data, axis=1)
        skewness = stats.skew(data, axis=1)
        kurtosis = stats.kurtosis(data, axis=1)
        features = np.array(
            [
                mean, std, variance, entropy, skewness, kurtosis
            ]
        ).T
        node_features = torch.from_numpy(features)
        return node_features
    
    
class CWTExtractor(BaseNodeExtractor):
    
    def __init__(self) -> None:
        super().__init__()
        
        
    def extract_features(self, data):
        node_features = torch.from_numpy(data)
        return node_features
    
    
    
class PSDExtractor(BaseNodeExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.bands = {
            "delta": [1, 4],
            "theta": [4, 7.5],
            "alpha": [7.5, 13],
            "lower_beta": [13, 16],
            "higher_beta": [16, 30], 
            "gamma": [30, 40]
        }
        
    def _bandpower(self, data, sf, band, window_sec=None, relative=True):
        """Compute the average power of the signal x in a specific frequency band.
        Parameters
        ----------
        data : 1d-array
            Input signal in the time-domain.
        sf : float
            Sampling frequency of the data.
        band : list
            Lower and upper frequencies of the band of interest.
        window_sec : float
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative : boolean
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.

        Return
        ------
        bp : float
            Absolute or relative band power.
        
        This function its taken from: https://raphaelvallat.com/bandpower.html
        """
        band = np.asarray(band)
        low, high = band

        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        # Compute the modified periodogram (Welch)
        freqs, psd = welch(data, sf, nperseg=nperseg, nfft=2048)

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]

        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using Simpson's rule.
        bp = simps(psd[idx_band], dx=freq_res)

        if relative:
            bp /= simps(psd, dx=freq_res)
        
        
        return bp
        
    def extract_features(self, data):
        psds = [[self._bandpower(channel, SF, band) for band in self.bands.values()] for channel in data]
        #psds = torch.tensor(psds)
        return psds
    