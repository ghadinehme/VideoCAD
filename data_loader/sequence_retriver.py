from abc import ABC, abstractmethod
import bisect
import os
import pickle
import numpy as np
from tqdm import tqdm

from functools import lru_cache


class SequenceRetriever(ABC):
    def __init__(self, data_files, image_files):
        self.data_files = data_files
        self.image_files = image_files

    @abstractmethod
    def get_sequence(self, idx):
        pass
    
    
    @abstractmethod
    def __len__(self):
        pass

    
class BaseSequenceRetriever(SequenceRetriever): 

    def get_sequence(self, idx):
        data = pickle.load(open(self.data_files[idx], 'rb'))
        frames = data['frames']
        actions = data['actions']
        base_file_id = os.path.basename(self.image_files[idx]).split("_")[0]
        return frames, actions, base_file_id
    
    def __len__(self):
        return len(self.data_files)
    