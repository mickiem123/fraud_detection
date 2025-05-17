import os
import sys 
import pandas as pd
import numpy as np 
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import setup_logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

logger = setup_logger()
@dataclass
class PreprocessorConfig():
    """
    Store col description here
    """
    windows_size = [1,7,30]
    delay = 7

class Preprocessor():
    def __init__(self, df: pd.DataFrame):
        # Ensure TX_DATETIME is datetime type
        
        self.df = df.copy()
        if not np.issubdtype(self.df["TX_DATETIME"].dtype, np.datetime64):
            self.df["TX_DATETIME"] = pd.to_datetime(self.df["TX_DATETIME"])
        self.config = PreprocessorConfig()

    def get_TX_DURING_WEEKEND(self):
        try:
            self.df["TX_DURING_WEEKEND"] = self.df["TX_DATETIME"].apply(lambda x: 1 if x.weekday() >= 5 else 0)
        except Exception as e:
            pass

    def get_TX_DURING_NIGHT(self):
        self.df["TX_DURING_NIGHT"] = self.df["TX_DATETIME"].apply(lambda x: 1 if 0 < x.hour <= 6 else 0)

    def get_CUSTOMER_ID_characteristic(self):
        
        try:
            logger.info("Initilizing get_CUSTOMER_ID_characteristic")
            for size in self.config.windows_size:
                logger.info(f"Starting for {size}d_WINDOW")

                self.df.sort_values("TX_DATETIME",inplace=True)
                logger.info("Calculating WINDOW_TX_COUNT")
                WINDOW_TX_COUNT = (
                    self.df.groupby("CUSTOMER_ID")
                    .rolling(f"{size}d", on="TX_DATETIME")["TRANSACTION_ID"].count()
                    .reset_index()
                    .rename(columns={"TRANSACTION_ID": f"CUSTOMER_ID_NB_TX_{size}DAY_WINDOW"})
                )
                logger.info("Calculating WINDOW_TX_MEAN")
                WINDOW_TX_MEAN = (
                    self.df.groupby("CUSTOMER_ID")
                    .rolling(f"{size}d", on="TX_DATETIME")["TX_AMOUNT"].mean()
                    .reset_index()
                    .rename(columns={"TX_AMOUNT": f"CUSTOMER_ID_AVG_AMOUNT_{size}DAY_WINDOW"})
                )

                logger.info("Merging WINDOW_TX_COUNT into self.df")
                self.df[[ "CUSTOMER_ID","TX_DATETIME", f"CUSTOMER_ID_NB_TX_{size}DAY_WINDOW"]] = WINDOW_TX_COUNT
                logger.info("Merging WINDOW_TX_MEAN into self.df")
                self.df[["CUSTOMER_ID","TX_DATETIME", f"CUSTOMER_ID_AVG_AMOUNT_{size}DAY_WINDOW"]] = WINDOW_TX_MEAN
                self.df['TX_DATETIME'] = pd.to_datetime(self.df['TX_DATETIME'])
        except Exception as e:
            raise CustomException(e, sys)

    def get_TERMINAL_ID_characteristic(self):
        try:
            logger.info("Initializing get_TERMINAL_ID_characteristic")
            logger.info("Calculating FRAUD_DELAY and TX_DELAY")
            FRAUD_DELAY = self.df.groupby("TERMINAL_ID").rolling(
            f"{self.config.delay}D", on="TX_DATETIME"
            )["TX_FRAUD"].sum()
            TX_DELAY = self.df.groupby("TERMINAL_ID").rolling(
            f"{self.config.delay}D", on="TX_DATETIME"
            )["TX_FRAUD"].count()
            for size in self.config.windows_size:
                logger.info(f"Calculating rolling sums/counts for {size}d_WINDOW + delay")
                self.df.sort_values(["TERMINAL_ID","TX_DATETIME"],inplace= True)
                logger.info("Sorted values for TERMINAL_ID,TX_DATETIME]")
                FRAUD_DELAY_WINDOW = self.df.groupby("TERMINAL_ID").rolling(
                    f"{size + self.config.delay}D", on="TX_DATETIME"
                )["TX_FRAUD"].sum()
                TX_DELAY_WINDOW = self.df.groupby("TERMINAL_ID").rolling(
                    f"{size + self.config.delay}D", on="TX_DATETIME"
                )["TX_FRAUD"].count()

                logger.info(f"Calculating FRAUD_WINDOW, TX_WINDOW, and RISK_WINDOW for {size}d_WINDOW")
                FRAUD_WINDOW = FRAUD_DELAY_WINDOW - FRAUD_DELAY
                TX_WINDOW = TX_DELAY_WINDOW - TX_DELAY
                RISK_WINDOW = FRAUD_WINDOW / TX_WINDOW
                
                logger.info(f"Inserting TX_WINDOW and RISK_WINDOW to self.df for {size}d_WINDOW")
                self.df.sort_values(["TERMINAL_ID","TX_DATETIME"],inplace= True)
                logger.info("Sorted values for TERMINAL_ID,TX_DATETIME]")
                self.df[["TERMINAL_ID","TX_DATETIME",f'TERMINAL_ID_NB_TX_{size}DAY_WINDOW']] = TX_WINDOW.reset_index().fillna(0)
                self.df[["TERMINAL_ID","TX_DATETIME",f'TERMINAL_ID_RISK_{size}DAY_WINDOW']]= RISK_WINDOW.reset_index().fillna(0)

        except Exception as e:
            logger.error(f"Error in get_TERMINAL_ID_characteristic: {e}")
            raise CustomException(e, sys)
            
def main():
    # Example: Load a sample file and run the terminal ID characteristic function
    df = pd.DataFrame()
    for file in os.listdir(rf"C:\Users\thuhi\workspace\fraud_detection\data\raw_data"):
        temp = pd.read_pickle(os.path.join(rf"C:\Users\thuhi\workspace\fraud_detection\data\raw_data", file))
        df = pd.concat([df, temp])

    preprocessor = Preprocessor(df)
    preprocessor.get_CUSTOMER_ID_characteristic()
    print(preprocessor.df)

if __name__ == "__main__":
    main()
