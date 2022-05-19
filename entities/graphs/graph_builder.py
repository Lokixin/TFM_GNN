import torch
from itertools import product
from abc import ABC, abstractmethod
from torch_geometric.data import Data
from .node_extractors import RawExtractor, StadisticalMomentsExtractor
from .edge_extractors import PearsonExtractor
from .constants import NUM_CHANNELS


class BaseGraphBuilder(ABC):
    
    @abstractmethod
    def __init__(self) -> None: 
        self.node_feature_extractor = None
        self.edge_feature_extractor = None
        
    
    @abstractmethod
    def build(self, data, label):
        node_features = self.node_feature_extractor.extract_features(data)
        edge_features = self.edge_feature_extractor.extract_features(data)
        format_label = self._format_label(label)
        edge_index = torch.tensor(
            [[a, b] for a, b in product(range(NUM_CHANNELS), range(NUM_CHANNELS))]
        ).t().contiguous()
        
        graph = Data(
            x=node_features,
            edge_attr=edge_features,
            label=format_label,
            edge_index=edge_index
        )
        
        return graph
    
    def _onehot_label(self, label)-> torch.Tensor:
        if label == "AD":
            return torch.tensor([[1, 0, 0]], dtype=torch.float64)
        if label == "HC":
            return torch.tensor([[0, 1, 0]], dtype=torch.float64)
        if label == "MCI":
            return torch.tensor([[0, 0, 1]], dtype=torch.float64)
        
    def _format_label(self, label) -> torch.Tensor:
        label_dict = {"AD": 0, "HC": 1, "MCI": 2}
        return label_dict[label]
        
            
class RawAndPearson(BaseGraphBuilder):
    def __init__(self) -> None:
        self.node_feature_extractor = RawExtractor()
        self.edge_feature_extractor = PearsonExtractor()
        
        
    def build(self, data, label):
        return super().build(data, label)
    

class MomentsAndPearson(BaseGraphBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.node_feature_extractor = StadisticalMomentsExtractor()
        self.edge_feature_extractor = PearsonExtractor(th=0.1)
        
    def build(self, data, label):
        return super().build(data, label)
    
    
