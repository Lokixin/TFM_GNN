import torch
import numpy as np
from abc import ABC, abstractmethod
from scipy import stats


class BaseNodeExtractor(ABC):
    
    @abstractmethod
    def __init__(self) -> None: ...
    
    @abstractmethod
    def extract_features(self, data: np.ndarray): ...
    
    
class RawExtractor(BaseNodeExtractor):
    
    def __init__(self) -> None:
        super().__init__()
        
        
    def extract_features(self, data):
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