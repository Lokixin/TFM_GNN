from ast import Mod
import torch.nn as nn
from .modelsTypes import Model
from .models import MAGE, EEGConvNetMini, EEGConvNetMiniV2, EEGConvNetMiniV3, EEGGraphConvNet, EEGGraphConvNetLSTM, EEGGraphConvNetTemporal, EEGConvNetMiniV2Attention, MultiLevelConvNet

class ModelFactory:
    
    def __init__(self) -> None:
        pass
    
    def create(self, model_id, **kwargs) -> nn.Module:
        if model_id == Model.EEGGRAPHCONVNET:
            return EEGGraphConvNet()
        
        if model_id == Model.EEGGRAPHCONVNETLSTM:
            return EEGGraphConvNetLSTM(**kwargs)
        
        if model_id == Model.EEGCONVNETMINI:
            return EEGConvNetMini(**kwargs)
        
        if model_id == Model.EEGCONVNETMINIV2:
            return EEGConvNetMiniV2(**kwargs)
        
        if model_id == Model.EEGCONVNETMINIV3:
            return EEGConvNetMiniV3(**kwargs)
        
        if model_id == Model.EEGCONVNETMINIV2ATTN:
            return EEGConvNetMiniV2Attention(**kwargs)
        
        if model_id == Model.EEGGRAPHCONVNETTEMPORAL:
            return EEGGraphConvNetTemporal()
        
        if model_id == Model.MULTILEVEL:
            return MultiLevelConvNet()
        
        if model_id == Model.MAGE:
            return MAGE(**kwargs)