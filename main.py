import torch
import pandas as pd
from torch.utils.data import Dataset
from entities.graphs.graph_builder import RawAndPearson, MomentsAndPearson
from entities.graphs.data_reader import read_record
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


class BaseDataset(Dataset):
    def __init__(self, indices ,builder, transform=None, target_transform=None):
        self.indices = indices
        self.builder = builder
        self.transform = transform
        self.target_transform = target_transform
        
        

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        current_path = self.indices.iloc[idx]["path"]
        raw_data = read_record(current_path)
        label = self.indices.iloc[idx]["label"]
        data = self.builder.build(raw_data, label)
        
        return data
    
def main():
    PATH = "C:/Projects/TFM/dataset/AD_MCI_HC_WINDOWED"
    INDEX_PATH = "C:/Projects/TFM/dataset/AD_MCI_HC_WINDOWED/data.csv"
    indices = pd.read_csv(INDEX_PATH, index_col="Unnamed: 0")

    indices = indices.drop(indices[indices.label == "MCI"].index)

    indices_hc = indices[indices.label == 'HC'].sample(frac=0.4)
    indices_ad = indices[indices.label == 'AD']
    indices = pd.concat([indices_hc, indices_ad])

    train_data, test_data = train_test_split(indices, shuffle=True)

    builder = RawAndPearson(normalize_nodes=True, normalize_edges=True)
    #builder = MomentsAndPearson()

    train_dataset = BaseDataset(train_data, builder)
    test_dataset = BaseDataset(test_data, builder)
    _BATCH_SIZE = 256
    train_dataloader = DataLoader(train_dataset, batch_size=_BATCH_SIZE, shuffle=True)#sampler=weighted_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=_BATCH_SIZE, shuffle=True)
    for data in train_dataloader:
        print(data[0].edge_attr)
        print(next(iter(data[0]))[1].shape)
        break
    

if __name__ == "__main__":
    main()