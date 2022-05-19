import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseEdgeExtractor(ABC):
    
    @abstractmethod
    def __init__(self) -> None: ...
    
    @abstractmethod
    def extract_features(self, data) -> None: ...
    
    
class PearsonExtractor(BaseEdgeExtractor):
    def __init__(self, **kwargs) -> None:
        self.th = kwargs.get("th", None)
    
    def extract_features(self, data) -> None:
        corr_matrix = np.corrcoef(data)
        
        if self.th:
            corr_matrix = np.where(
                abs(corr_matrix) < self.th, 0, corr_matrix
            )
            
        corr_matrix = torch.from_numpy(corr_matrix)
        return corr_matrix