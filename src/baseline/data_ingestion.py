import os
import sys
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod
from src.utils import setup_logger
from src.exception import CustomException
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

logger = setup_logger()

@dataclass
class DataIngestorConfig():
    output_feature="TX_FRAUD"

    input_features=['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
       'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
       'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
       'TERMINAL_ID_RISK_30DAY_WINDOW']
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

        
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self,path)-> pd.DataFrame:
        pass

class DurationTrainTestDataIngestor(DataIngestor):
    def ingest(self,dir_path,*, start_train_date, train_duration,delay,test_duration,
               end_train_date = None,end_test_date = None,start_test_date = None) -> pd.DataFrame:
        try:
            logger.info(f"Calculating dates")

            if not end_train_date:
                end_train_date = pd.Timestamp(start_train_date) + pd.Timedelta(days=train_duration)
                
            if not start_test_date:
                start_test_date = pd.Timestamp(end_train_date) + pd.Timedelta(days=delay)

            if not end_test_date:
                end_test_date = pd.Timestamp(start_test_date) + pd.Timedelta(days=test_duration)
            
            logger.info("Preparing files list")
            train_file = []
            test_file =[]
            for file in os.listdir(dir_path):
                file,self.file_extension = os.path.splitext(file)
                

                if pd.Timestamp(start_train_date) <= pd.Timestamp(file)<= pd.Timestamp(end_train_date):
                    train_file.append(file)

                if pd.Timestamp(start_test_date) <= pd.Timestamp(file)<= pd.Timestamp(end_test_date):
                    test_file.append(file)        

            logger.info("Loading file into Dataframe")

            train_df = self.get_files(train_file)
            logger.info(f"{train_df}")
            test_df = self.get_files(test_file)
            
            if train_df is None or train_df.empty:
                logger.warning("train_df is empty or None")
            if test_df is None or test_df.empty:
                logger.warning("test_df is empty or None")

            return train_df,test_df
        except Exception as e:
            raise CustomException(e,sys)

            
    def get_files(self,files_list):

        files_list = pd.Series(files_list).map(lambda x: os.path.join("data","raw_data",f"{x}.pkl"))
        logger.info(files_list)
        df = pd.DataFrame()
        for file in files_list:
            df = pd.concat([df,pd.read_pickle(file)])
        return df

class FraudDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        'Initialization'
        self.config = DataIngestorConfig()
        print(self.config.input_features)
        self.x = torch.FloatTensor(df[self.config.input_features].values)
        self.y = torch.FloatTensor(df[self.config.output_feature].values)

    def __len__(self):
        'Returns the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        return (
            self.x[index].to(self.config.DEVICE),
            self.y[index].to(self.config.DEVICE)
        )

class DataIngestorFactory():
    @staticmethod
    def create_ingestor(data_type):
        try:
               
            if data_type == "duration_pkl":
                return DurationTrainTestDataIngestor()
            
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def cuda_intergrate(df_train,df_test):
        try:
            return FraudDataset(df_train),FraudDataset(df_test)
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    print(os.getcwd())
    factory = DataIngestorFactory()
    ingestor = factory.create_ingestor("duration_pkl")

    train_df,test_df = ingestor.ingest(
        rf"C:\Users\thuhi\workspace\fraud_detection\data\raw_data",
        start_train_date="2018-04-01",
        train_duration=1,
        delay=5,
        test_duration=1)
    train_df,test_df = factory.cuda_intergrate(train_df,test_df)
    print(train_df,test_df)
    #artifact_dir = os.path.join(os.getcwd(), "artifact")
    #os.makedirs(artifact_dir, exist_ok=True)

    #train_data.to_csv(os.path.join(artifact_dir, "train_data.csv"), index=False)
    #test_data.to_csv(os.path.join(artifact_dir, "test_data.csv"), index=False)
    
    #logger.info(f"Exported train and test data to {artifact_dir}")
    