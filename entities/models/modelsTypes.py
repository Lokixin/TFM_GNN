from enum import Enum, unique, auto


@unique
class Model(Enum):
    EEGGRAPHCONVNET = auto()
    EEGGRAPHCONVNETLSTM = auto()
    EEGSMALL = auto()
    EEGCONVNETMINI = auto()
    EEGCONVNETMINIV2 = auto()
    EEGCONVNETMINIV3 = auto()
    EEGCONVNETMINILSTM = auto()
    