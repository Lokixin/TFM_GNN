import torch
import numpy as np
from abc import ABC, abstractmethod

from .preprocessing import normalize


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
        
        corr_matrix = np.where(
            corr_matrix == 1, 0, corr_matrix
        )
        
        if self.th:
            corr_matrix = np.where(
                abs(corr_matrix) < self.th, 0, corr_matrix
            )
            
        corr_matrix = torch.from_numpy(corr_matrix)
        return corr_matrix