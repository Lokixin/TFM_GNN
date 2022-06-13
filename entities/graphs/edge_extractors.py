import imp
import torch
import numpy as np
from abc import ABC, abstractmethod
from scipy.signal import csd, hilbert, welch

from .preprocessing import normalize
from .constants import SF


class BaseEdgeExtractor(ABC):
    
    @abstractmethod
    def __init__(self) -> None: ...
    
    @abstractmethod
    def extract_features(self, data) -> None: ...
        
    
    
class PearsonExtractor(BaseEdgeExtractor):
    def __init__(self, **kwargs) -> None:
        self.th = kwargs.get("th", None)
        self.normalize = kwargs.get("normalize", False)
    
    def extract_features(self, data) -> None:
        if self.normalize:
            data = normalize(data)
            
        corr_matrix = np.corrcoef(data)
        
        # Set to 0 values undes a
        if self.th:
            corr_matrix = np.where(
                abs(corr_matrix) < self.th, 0, corr_matrix
            )
        
        # Set to 0 self-loops
        corr_matrix = np.where(
            corr_matrix == 1, 0, corr_matrix
        )
        
        #corr_matrix = torch.from_numpy(corr_matrix)
        return corr_matrix
    
    
class PLIExtractor(BaseEdgeExtractor):
    def __init__(self, **kwargs) -> None:
        self.th = kwargs.get("th", None)
        self.normalize = kwargs.get("normalize", False)
        self.fs = kwargs.get("fs", 256)
        
        
    
    def extract_features(self, data) -> None:
        rows, cols = data.shape
        PLImatrix = np.zeros((rows, rows))    
        
        hilbert_transform = np.imag(hilbert(data))
        
        if self.normalize:
            data = normalize(data)
            
        for row in range(rows):
            for col in range(rows):
                PLImatrix[row, col] = np.abs(np.mean(np.exp(1j * (hilbert_transform[row] - hilbert_transform[col]))))  
          
        # Set to 0 values undes a th
        if self.th:
            PLImatrix = np.where(
                abs(PLImatrix) < self.th, 0, PLImatrix
            )
        
        # Remove self loops
        PLImatrix = np.where(
                abs(PLImatrix) == 1.0, 0, PLImatrix
            )
        
        #PLImatrix = torch.from_numpy(PLImatrix)
        return PLImatrix
    
    
class SpectralCoherenceExtractor(BaseEdgeExtractor):
    def __init__(self, **kwargs) -> None:
        self.th = kwargs.get("th", None)
        self.normalize = kwargs.get("normalize", False)
        
    
    def _compute_coherence(ch1, ch2, fs=SF):
        _, cross_spectral = csd(ch1, ch2,  fs=SF)
        coherence = np.abs(np.mean(cross_spectral)) / np.sqrt(np.mean(welch(ch1)) * np.mean(welch(ch2)))
        return coherence
    
    def extract_features(self, data) -> None:
        if self.normalize:
            data = normalize(data)
            
        
        #coherence = [[np.abs(np.mean(csd(ch1, ch2, fs=256)[1])) / np.sqrt(np.mean(welch(ch1)) * np.mean(welch(ch2))) for ch2 in data] for ch1 in data]#[[ self._compute_coherence(channel1, channel2) for channel2 in data ] for channel1 in data]
        coherence = [[np.abs(np.mean(csd(ch1, ch2, fs=256)[1]))  for ch2 in data] for ch1 in data]
        # Set to 0 values undes a
        if self.th:
            coherence = np.where(
                abs(coherence) < self.th, 0, coherence
            )
        
        # Set to 0 self-loops
        coherence = np.where(
            coherence == 1, 0, coherence
        )
        
        #coherence = torch.tensor(coherence)
        return coherence