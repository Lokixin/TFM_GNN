
import sys
import os
import pathlib
import scipy.io as sio

from entities.graphs.data_reader import read_record
from entities.graphs.edge_extractors import PearsonExtractor, PLIExtractor, SpectralCoherenceExtractor
from entities.graphs.node_extractors import PSDExtractor




class RecordMatSaver:
    def __init__(self) -> None:
        pass
    
    def save(self, save_path, matrix):
        sio.savemat(save_path, {"EEG": matrix})


class DatasetIterator:
    
    def __init__(self, root_path, save_path, extractor) -> None:
        self.root_path = pathlib.Path(root_path).resolve()
        self.save_path = pathlib.Path(save_path).resolve()
        self.extractor = extractor
        self.saver = RecordMatSaver()
    
    
    def iterate(self, verbose=False):
        class_folders = [ self.root_path.joinpath(folder) for folder in os.listdir(self.root_path) ]
        
        for subfolder in class_folders:
            
            if not subfolder.is_dir():
                continue
            
            for record in os.listdir(subfolder):
                
                if not ".mat" in record: 
                    continue
                
                matrix = read_record(str(subfolder.joinpath(record)))
                processed_matrix = self.extractor.extract_features(matrix)
                
                new_path = self.save_path.joinpath(subfolder.stem, record)
                
                self.saver.save(new_path, processed_matrix)
                if verbose:
                    print(f"Saving into: {new_path}")
                
                
                

if __name__ == "__main__":
    PATH = "C:/Projects/TFM/dataset/AD_MCI_HC_WINDOWED"
    NEW_PATH = "C:/Projects/TFM/dataset/AD_MCI_HC_PEARSON"

    extractors = {NEW_PATH: PearsonExtractor()}
    
    for new_path, extractor in extractors.items():
        dataset_generator = DatasetIterator(PATH, new_path, extractor)
        dataset_generator.iterate(verbose=True)
    
    
    