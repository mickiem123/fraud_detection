import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from src.baseline.data_ingestion import DataIngestorFactory,DataIngestorConfig
import os
import sys

class FraudDetectionDataset(Dataset):
    """
    A custom PyTorch Dataset for fraud detection tasks.
    This dataset wraps a pandas DataFrame and provides access to feature and target tensors
    for use in neural network models. The features and target columns are determined by the
    DataIngestorConfig configuration.
    Args:
        df (pandas.DataFrame): The input DataFrame containing both features and target columns.
        mode (str): transformed or none
    Attributes:
        config (DataIngestorConfig): Configuration object specifying input and output feature columns.
        df (pandas.DataFrame): The original DataFrame.
        features (pandas.DataFrame): DataFrame containing feature columns.
        target (pandas.Series or pandas.DataFrame): Series or DataFrame containing the target variable.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns a tuple (features, target) for the given index, both as torch.float32 tensors.
    """

    def __init__(self,df,mode=None):
        self.config = DataIngestorConfig()
        
        self.df = df
        if mode == "transformed":
            self.features = df.loc[:,self.config.input_features_transformed]
        else:
             self.features = df.loc[:,self.config.input_features]
        self.target = df[self.config.output_feature]

    def __len__(self):
        return len(self.df)
    def debug(self):
        print(self.df)
        print(self.features)
        print(self.features.columns)
    def __getitem__(self, index):
        features = torch.tensor(self.features.iloc[index],dtype=torch.float32)
        target = torch.tensor(self.target.iloc[index],dtype=torch.int8)
        return features,target


        
    
