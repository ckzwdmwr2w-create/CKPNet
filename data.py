# coding:utf-8
import csv, os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class EssayDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        return self.dataset[idx]
    
