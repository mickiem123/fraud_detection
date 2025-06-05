import sys
import os
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import lightning as L
import torch.nn.functional as F
from torchmetrics import AveragePrecision, AUROC, Precision, Recall
from src.components.nn_data_ingestion import FraudDetectionDataset
from src.baseline.data_ingestion import DataIngestorFactory, DataIngestorConfig
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # Updated import
from src.utils import setup_logger, seed_everything
from src.exception import CustomException
from src.baseline.features_engineering import PreprocessorPipeline

import warnings
warnings.filterwarnings("ignore")

logger = setup_logger()
seed_everything()

class Feed_Forward_NN(L.LightningModule):
    def __init__(self, input_size, hidden_size,pos_weight ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.learning_rate = 0.001
        self.ap = AveragePrecision(task="binary")
        self.aucroc = AUROC(task="binary")
        self.top_100 = Precision(task="binary", top_k=100)
        self.recall = Recall(task="binary")  # Updated to use Recall with task="binary"
        self.precision = Precision(task="binary") 
        self.pos_weight = pos_weight
       # Updated to use Precision with task="binary"

    def focal_loss(self, inputs, targets, alpha, gamma):
        """
        Compute Focal Loss for binary classification.
        
        Args:
            inputs (torch.Tensor): Raw model outputs (logits).
            targets (torch.Tensor): True labels (0 or 1).
            
        Returns:
            torch.Tensor: Computed focal loss (unreduced).
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return (alpha * (1 - pt) ** gamma * BCE_loss).sum()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        try:
            x, y = batch
            x = x.view(x.size(0), -1)
            y_hat = self.model(x).view(-1)
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float(),pos_weight=self.pos_weight)
            probs = torch.sigmoid(y_hat)
            f1 = 2 * (self.precision(probs, y) * self.recall(probs, y)) / (self.precision(probs, y) + self.recall(probs, y) + 1e-8)
            auc_roc = self.aucroc(probs, y) if y.sum() > 0 else torch.tensor(0.0, device=self.device)
            recall = self.recall(probs, y) if y.sum() > 0 else torch.tensor(0.0, device=self.device)
            precision = self.precision(probs, y) if y.sum() > 0 else torch.tensor(0.0, device=self.device)
            self.log_dict({
                "train_loss": loss,
                "train_auc_roc": auc_roc,
                "train_recall": recall,
                "train_precision": precision,
                "train_f1": f1,
                "train_avg_prob": probs.mean()
            }, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        except Exception as e:
            raise CustomException(e, sys)
    def on_after_backward(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.log(f"grad_{name}", param.grad.abs().mean(), on_step=True, on_epoch=False)
    def validation_step(self, batch, batch_idx):
        try:
            x, y = batch
            x = x.view(x.size(0), -1)
            y_hat = self.model(x).view(-1)
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float(),pos_weight=self.pos_weight)
            probs = torch.sigmoid(y_hat)
            auc_roc = self.aucroc(probs, y) if y.sum() > 0 else torch.tensor(0.0, device=self.device)
            recall = self.recall(probs, y) if y.sum() > 0 else torch.tensor(0.0, device=self.device)
            precision = self.precision(probs, y) if y.sum() > 0 else torch.tensor(0.0, device=self.device)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            self.log_dict({
                "val_loss": loss,
                "val_auc_roc": auc_roc,
                "val_recall": recall,
                "val_precision": precision,
                "val_f1": f1,
                "val_avg_prob": probs.mean()
            }, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        except Exception as e:
            raise CustomException(e, sys)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    
    batch_size = 2048
    epoch = 40
    patience = 15
    pos_mul = 0.6
    seed_everything(42)
    
    factory = DataIngestorFactory()
    ingestor = factory.create_ingestor("duration_pkl")
    train_df, validation_df = ingestor.ingest(
        dir_path=rf"C:\Users\thuhi\workspace\fraud_detection\data\raw_data",
        start_train_date="2018-08-01",
        train_duration=7,
        test_duration=7,
        delay=7
    )

    train_preprocessed = PreprocessorPipeline(train_df).process()
    validation_preprocessed = PreprocessorPipeline(validation_df).process()
    pos_weight = pos_mul * 1/torch.tensor(train_preprocessed[DataIngestorConfig().output_feature].sum()
                              / (len(train_preprocessed) - train_preprocessed[DataIngestorConfig().output_feature].sum()))
    pos = train_df[DataIngestorConfig().output_feature].sum()
    pos_processed = train_preprocessed[DataIngestorConfig().output_feature].sum()
    logger.info(f"total pos: {pos}, total pos after process: {pos_processed}")
    logger.info(f"total data: {len(train_df)}, total data after process: {len(train_preprocessed)}")
    logger.info(pos_weight)
    train_data = FraudDetectionDataset(train_preprocessed, mode="transformed")
    validation_data = FraudDetectionDataset(validation_preprocessed, mode="transformed")
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        num_workers=8,
        persistent_workers=True,
        shuffle=False

    )

    run = 1
    log_dir = os.path.join("log", "tensor_log")
    os.makedirs(log_dir, exist_ok=True)
    while True:
        version = f"run_{run}"
        if version in os.listdir(log_dir):
            run += 1
        else:
            logger = TensorBoardLogger(save_dir="log/", name="tensor_log", version=version)
            break

    model = Feed_Forward_NN(input_size=len(train_data.config.input_features_transformed), hidden_size=512,pos_weight=pos_weight)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best-model-{epoch:02d}-{val_loss:.2f}-{pos_weight:.2f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True
    )

    trainer = L.Trainer(
        fast_dev_run=False,
        num_sanity_val_steps=-1,
        max_epochs=epoch,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=checkpoint_callback
        
    )
    
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )