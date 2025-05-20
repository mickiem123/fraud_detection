import os
import sys
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from src.utils import setup_logger
from src.exception import CustomException

logger = setup_logger()


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self,path,):
        pass

class DurationTrainTestDataIngestor(DataIngestor):
    def ingest(self,dir_path,*, start_train_date, train_duration,delay,test_duration,
               end_train_date = None,end_test_date = None,start_test_date = None):
        logger.info(f"Calculating dates")

        if not end_train_date:
            end_train_date = pd.Timestamp(start_train_date) + pd.Timedelta(days=train_duration)
            
        if not start_test_date:
            start_test_date = pd.Timestamp(end_train_date) + pd.Timedelta(days=delay)

        if not end_test_date:
            end_test_date = pd.Timestamp(start_test_date) + pd.Timedelta(days=test_duration)
        
        logger.info("Preparing files list")

        for file in os.listdir(dir_path):
            file = os.path.splitext(file)[0]
            train_file = []
            test_file =[]

            if pd.Timestamp(start_train_date) <= pd.Timestamp(file)<= pd.Timestamp(end_train_date):
                train_file.append(file)
            if pd.Timestamp(start_test_date) <= pd.Timestamp(file)<= pd.Timestamp(end_test_date):
                test_file.append(file)        

        logger.info("Loading file into Dataframe")

        train_df = self.get_files(train_file)
        test_df = self.get_files(test_file)

        return train_df,test_df

        
    def get_files(self,files_list):
        df = pd.DataFrame()
        for file in files_list:
            df = pd.concat(df,pd.read_csv(file))
        return df
    
class DataIngestorFactory():
    @staticmethod
    def create_ingestor(data_type):
        try:
            if data_type == "duration_csv":
                return DurationTrainTestDataIngestor()
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    print(os.getcwd())
    factory = DataIngestorFactory()
    ingestor = factory.create_ingestor("duration_csv")
    ingestor.ingest(
        rf"C:\Users\thuhi\workspace\fraud_detection\simulated-data-raw\raw",
        start_train_date="2023-01-01",
        train_duration=7,
        delay=5,
        test_duration=10)
            