import torch.nn as nn
from .modelsTypes import Model
from .models import EEGGraphConvNet

class ModelFactory:
    
    def __init__(self) -> None:
        pass
    
    def create(self, model_id) -> nn.Module:
        if model_id == Model.EEGGRAPHCONVNET:
            return EEGGraphConvNet()