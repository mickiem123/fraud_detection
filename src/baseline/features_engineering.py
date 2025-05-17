import os
import sys 
import pandas as pd
import numpy as np 
from dataclasses import dataclass
from src.exception import CustomException
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

@dataclass
class PreprocessorConfig():
    """
    Store col description here
    """
    windows_size = [1,7,30]
    delay = 7
class Preprocessor():
    def __init__(self,df:pd.DataFrame):
        self.df = df
        self.config = PreprocessorConfig()

    def get_TX_DURING_WEEKEND(self):
        try:
            self.df["TX_DURING_WEEKEND"] = self.df["TX_DATETIME"].map(lambda x: 1 if x.weekday()>=5 else 0)
            return self.df
        except Exception as e:
            pass
    def get_TX_DURING_NIGHT(self):
        self.df["TX_DURING_NIGHT"] = self.df["TX_DATETIME"].map(lambda x: 1 if 0< x.hour() <= 6 else 0)
    
    def get_CUSTOMER_ID_characteristic (self):
        try:
            for size in self.config.windows_size:

                self.df[f"CUSTOMER_ID_NB_TX_{size}DAY_WINDOW"] = (
                    self.df.groupby("CUSTOMER_ID")
                    .rolling(f"{size}D", on="TX_DATETIME")
                    ['TRANSACTION_ID'].count()
                    .reset_index(drop=True)
                    )
                self.df[f'CUSTOMER_ID_AVG_AMOUNT_{size}DAY_WINDOW'] = (
                    self.df.groupby("CUSTOMER_ID")
                    .rolling(f"{size}d",on="TX_DATETIME")
                    ["TX_AMOUNT"].mean()
                )
            return self.df
        except Exception as e:
            raise CustomException(e,sys)
            
    def get_TERMINAL_ID_characteristic(self):
        try:
            FRAUD_DELAY  = self.df.groupby("TERMINAL_ID").rolling(f"{self.config.delay}d", on="TX_DATETIME")["TX_FRAUD"].sum()
            TX_DELAY = self.df.groupby("TERMINAL_ID").rolling(f"{self.config.delay}d", on = "TX_DATETIME")["TX_FRAUD"].count()
            for size in self.config.windows_size:
                FRAUD_DELAY_WINDOW = self.df.groupby("TERMINAL_ID").rolling(
                    f"{size + self.config.delay}d",
                    on="TX_DATETIME")["TX_FRAUD"].sum()
                TX_DELAY_WINDOW = self.df.groupby("TERMINAL_ID").rolling(
                    f"{size + self.config.delay}d",
                    on="TX_DATETIME")["TX_FRAUD"].count()

                FRAUD_WINDOW = FRAUD_DELAY_WINDOW - FRAUD_DELAY
                TX_WINDOW = TX_DELAY_WINDOW - TX_DELAY 

                RISK_WINDOW = FRAUD_WINDOW /TX_WINDOW

                self.df['TERMINAL_ID_NB_TX_'+str(size)+'DAY_WINDOW']=list(TX_WINDOW)
                self.df['TERMINAL_ID_RISK_'+str(size)+'DAY_WINDOW']=list(RISK_WINDOW)
                self.df['TERMINAL_ID_NB_TX_'+str(size)+'DAY_WINDOW'] = self.df['TERMINAL_ID_NB_TX_'+str(size)+'DAY_WINDOW'].fillna(0)
                self.df['TERMINAL_ID_RISK_'+str(size)+'DAY_WINDOW'] = self.df['TERMINAL_ID_RISK_'+str(size)+'DAY_WINDOW'].fillna(0)
        except Exception as e:
            raise CustomException(e,sys)
            
def main():
    # Example: Load a sample file and run the terminal ID characteristic function
    import pandas as pd
    import os
    data_path = os.path.join(os.path.dirname(__file__), '../../data/raw_data/2018-04-01.pkl')
    df = pd.read_pickle(data_path)
    preprocessor = Preprocessor(df)
    preprocessor.get_TERMINAL_ID_characteristic()
    preprocessor.get_CUSTOMER_ID_characteristic()
    print(preprocessor.df.head())

if __name__ == "__main__":
    main()



