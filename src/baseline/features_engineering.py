import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from src.exception import CustomException
from src.utils import setup_logger
from src.components.data_ingestion import DataIngestorFactory
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

logger = setup_logger()

# Configuration class
@dataclass
class PreprocessorConfig:
    """Configuration for preprocessing steps."""
    windows_size = [1, 7, 30]
    delay = 7

# Abstract base class for processing steps
class ProcessingStep(ABC):
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a processing step to the DataFrame and return the modified DataFrame."""
        pass

# Concrete step for TX_DURING_WEEKEND
class WeekendStep(ProcessingStep):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting WeekendStep")
        try:
            df["TX_DURING_WEEKEND"] = df["TX_DATETIME"].apply(lambda x: 1 if x.weekday() >= 5 else 0)
            return df
        except Exception as e:
            logger.error(f"Error in WeekendStep: {e}")
            raise CustomException(e, sys)

# Concrete step for TX_DURING_NIGHT
class NightStep(ProcessingStep):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting NightStep")
        try:
            df["TX_DURING_NIGHT"] = df["TX_DATETIME"].apply(lambda x: 1 if 0 < x.hour <= 6 else 0)
            return df
        except Exception as e:
            logger.error(f"Error in NightStep: {e}")
            raise CustomException(e, sys)

# Concrete step for CUSTOMER_ID_characteristic
class CustomerCharacteristicStep(ProcessingStep):
    def __init__(self, config: PreprocessorConfig):
        self.config = config

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting CustomerCharacteristicStep")
        try:
            logger.info("Initializing CustomerCharacteristicStep")
            for size in self.config.windows_size:
                logger.info(f"Starting for {size}d_WINDOW")
                df.sort_values("TX_DATETIME", inplace=True)
                
                logger.info("Calculating WINDOW_TX_COUNT")
                WINDOW_TX_COUNT = (
                    df.groupby("CUSTOMER_ID")
                    .rolling(f"{size}d", on="TX_DATETIME")["TRANSACTION_ID"].count()
                    .reset_index()
                    .rename(columns={"TRANSACTION_ID": f"CUSTOMER_ID_NB_TX_{size}DAY_WINDOW"})
                )
                
                logger.info("Calculating WINDOW_TX_MEAN")
                WINDOW_TX_MEAN = (
                    df.groupby("CUSTOMER_ID")
                    .rolling(f"{size}d", on="TX_DATETIME")["TX_AMOUNT"].mean()
                    .reset_index()
                    .rename(columns={"TX_AMOUNT": f"CUSTOMER_ID_AVG_AMOUNT_{size}DAY_WINDOW"})
                )
                
                logger.info("Merging WINDOW_TX_COUNT and WINDOW_TX_MEAN into df")
                df[f"CUSTOMER_ID_NB_TX_{size}DAY_WINDOW"] = WINDOW_TX_COUNT[f"CUSTOMER_ID_NB_TX_{size}DAY_WINDOW"]
                df[f"CUSTOMER_ID_AVG_AMOUNT_{size}DAY_WINDOW"] = WINDOW_TX_MEAN[f"CUSTOMER_ID_AVG_AMOUNT_{size}DAY_WINDOW"]
            return df
        except Exception as e:
            logger.error(f"Error in CustomerCharacteristicStep: {e}")
            raise CustomException(e, sys)

# Concrete step for TERMINAL_ID_characteristic
class TerminalCharacteristicStep(ProcessingStep):
    def __init__(self, config: PreprocessorConfig):
        self.config = config

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting TerminalCharacteristicStep")
        try:
            logger.info("Initializing TerminalCharacteristicStep")
            logger.info("Calculating FRAUD_DELAY and TX_DELAY")
            FRAUD_DELAY = df.groupby("TERMINAL_ID").rolling(
                f"{self.config.delay}D", on="TX_DATETIME"
            )["TX_FRAUD"].sum()
            TX_DELAY = df.groupby("TERMINAL_ID").rolling(
                f"{self.config.delay}D", on="TX_DATETIME"
            )["TX_FRAUD"].count()

            for size in self.config.windows_size:
                logger.info(f"Calculating rolling sums/counts for {size}d_WINDOW + delay")
                df.sort_values(["TERMINAL_ID", "TX_DATETIME"], inplace=True)
                
                FRAUD_DELAY_WINDOW = df.groupby("TERMINAL_ID").rolling(
                    f"{size + self.config.delay}D", on="TX_DATETIME"
                )["TX_FRAUD"].sum()
                TX_DELAY_WINDOW = df.groupby("TERMINAL_ID").rolling(
                    f"{size + self.config.delay}D", on="TX_DATETIME"
                )["TX_FRAUD"].count()

                logger.info(f"Calculating FRAUD_WINDOW, TX_WINDOW, and RISK_WINDOW for {size}d_WINDOW")
                FRAUD_WINDOW = FRAUD_DELAY_WINDOW - FRAUD_DELAY
                TX_WINDOW = TX_DELAY_WINDOW - TX_DELAY
                RISK_WINDOW = FRAUD_WINDOW / TX_WINDOW
                
                logger.info(f"Inserting TX_WINDOW and RISK_WINDOW to df for {size}d_WINDOW")
                df[f"TERMINAL_ID_NB_TX_{size}DAY_WINDOW"] = TX_WINDOW.reset_index(drop=True).fillna(0)
                df[f"TERMINAL_ID_RISK_{size}DAY_WINDOW"] = RISK_WINDOW.reset_index(drop=True).fillna(0)
            return df
        except Exception as e:
            logger.error(f"Error in TerminalCharacteristicStep: {e}")
            raise CustomException(e, sys)

# Pipeline class to manage and execute steps
class PreprocessorPipeline:
    def __init__(self, df: pd.DataFrame, config: PreprocessorConfig):
        logger.info("Initiating processing")
        self.df = df.copy()

        if not np.issubdtype(self.df["TX_DATETIME"].dtype, np.datetime64):
            self.df["TX_DATETIME"] = pd.to_datetime(self.df["TX_DATETIME"])
        self.config = config
        self.steps = [
            WeekendStep(),
            NightStep(),
            CustomerCharacteristicStep(self.config),
            TerminalCharacteristicStep(self.config)
        ]

    def process(self) -> pd.DataFrame:
        """Run all preprocessing steps in sequence."""
        current_df = self.df.copy()
        for step in self.steps:
            logger.info(f"Executing step: {step.__class__.__name__}")
            try:
                current_df = step.process(current_df)
            except Exception as e:
                logger.error(f"Exception in pipeline step {step.__class__.__name__}: {e}")
                raise CustomException(e, sys)
        self.df = current_df
        return self.df

# Main function
def main():
    factory = DataIngestorFactory()
    ingestor = factory.create_ingestor("duration_pkl")

    train_df, test_df = ingestor.ingest(
        rf"C:\Users\thuhi\workspace\fraud_detection\data\raw_data",
        start_train_date="2018-04-01",
        train_duration=7,
        delay=7,
        test_duration=7)

    config = PreprocessorConfig()
    for df in [train_df, test_df]:
        pipeline = PreprocessorPipeline(df, config)
        processed_df = pipeline.process()

        if df is train_df:
            output_path = os.path.join("artifact", "processed_train_data.csv")
        else:
            output_path = os.path.join("artifact", "processed_test_data.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        print(processed_df)

if __name__ == "__main__":
    main()