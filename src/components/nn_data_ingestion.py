import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from src.baseline.data_ingestion import DataIngestorFactory,DataIngestorConfig
from src.utils import setup_logger
import os
import pandas as pd
import numpy as np
import sys
import time

logger =setup_logger()
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

class SequentialFraudDetectionDataset(Dataset):
    def __init__(self,df:pd.DataFrame,seq_len =5,transformed = True):
        self.config = DataIngestorConfig()
        self.seq_len = seq_len
        self.num_samples = len(df)
        df.sort_values("TX_DATETIME",inplace=True)
        df.reset_index(drop=True,inplace=True)
        df["tmp_idx"] = np.arange(len(df))
        for i in range(1, seq_len+1):
            df[f"tx{i}"] = df.groupby("CUSTOMER_ID")["tmp_idx"].shift(seq_len-i)
            df = df.sort_values(["CUSTOMER_ID", "TX_DATETIME"]).fillna(self.num_samples)
        # Create a -1 index row with all zero values (matching df columns)
        zero_row = pd.DataFrame({col: [0] for col in df.columns}, index=[self.num_samples ])
        self.df = pd.concat([zero_row, df])
        # Precompute features and targets as tensors
        if transformed:
            self.features = torch.tensor(self.df[self.config.input_features_transformed].values, dtype=torch.float32)
        else:
            self.features = torch.tensor(self.df[self.config.input_features].values, dtype=torch.float32)
        self.targets = torch.tensor(self.df[self.config.output_feature].values, dtype=torch.int8)
        # Precompute sequence indices
        self.tx_indices = torch.tensor(self.df[[f"tx{i}" for i in range(1, seq_len + 1)]].values, dtype=torch.long)
    def __len__(self):
        return self.num_samples
    def __getitem__(self, index):
        # Use precomputed tensors with gathered indices
        #st = time.time()
        tx_ids = self.tx_indices[index]
        # Gather features for the sequence
        features = self.features[tx_ids]
        target = self.targets[index]
        #logger.info(f"time{time.time()-st}")
        return features, target
# In your nn_data_ingestion.py

class BaggingSequentialFraudDetectionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len=7):
        # NOTE: seq_len is the TOTAL number of transactions involved.
        # It will be split into (seq_len - 1) for history and 1 for the target.
        self.config = DataIngestorConfig()
        
        # We need seq_len-1 historical steps
        self.historical_len = seq_len - 1

        df = df.sort_values("TX_DATETIME").reset_index(drop=True)

        features_tensor = torch.tensor(df[self.config.input_features_transformed].values, dtype=torch.float32)
        self.targets = torch.tensor(df[self.config.output_feature].values, dtype=torch.int8)

        # The target transaction features are simply the original features tensor
        self.target_features = features_tensor

        # --- Create Historical Sequence Data ---
        padding_transaction = torch.zeros(1, features_tensor.shape[1], dtype=torch.float32)
        self.historical_features_pool = torch.vstack([features_tensor, padding_transaction])
        padding_idx = len(self.historical_features_pool) - 1

        df_for_indices = pd.DataFrame({
            'CUSTOMER_ID': df['CUSTOMER_ID'],
            'tmp_idx': np.arange(len(df))
        })
        
        df_groupby_customer = df_for_indices.groupby('CUSTOMER_ID')

        # This loop now correctly creates ONLY the historical sequence indices
        # It looks from shift(historical_len) down to shift(1).
        sequences = {
            f"tx_{i}": df_groupby_customer['tmp_idx'].shift(self.historical_len - i)
            for i in range(self.historical_len)
        }
        
        sequences_df = pd.DataFrame(sequences)
        self.historical_indices = torch.tensor(sequences_df.fillna(padding_idx).values, dtype=torch.long)
        
        self.num_samples = len(df)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get the indices for the historical part of the sequence
        hist_indices = self.historical_indices[index]
        
        # Assemble the historical sequence
        historical_sequence = self.historical_features_pool[hist_indices] # Shape: [hist_len, features]
        
        # Get the features for the target transaction
        target_transaction_features = self.target_features[index] # Shape: [features]
        
        # Get the label for the target transaction
        target_label = self.targets[index]
        
        return (historical_sequence, target_transaction_features), target_label
if __name__ == "__main__":
    factory = DataIngestorFactory()
    ingestor = factory.create_ingestor("duration_pkl")
    train_df, validation_df = ingestor.ingest(
        dir_path=rf"C:\Users\thuhi\workspace\fraud_detection\data\transformed_data",
        start_train_date="2018-05-15",
        train_duration=7,
        test_duration=7,
        delay=7
    )
    dataset = SequentialFraudDetectionDataset(train_df,seq_len=7)
    for i in range(100):
        dataset[i]