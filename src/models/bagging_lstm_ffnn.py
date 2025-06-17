import os
import torch.nn as nn
import torch
import lightning as L
import torch.nn.functional as F
from torchmetrics import AveragePrecision, Precision, Recall,AUROC
from src.components.nn_data_ingestion import BaggingSequentialFraudDetectionDataset

from src.components.data_ingestion import DataIngestorFactory, DataIngestorConfig
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # Updated import
from src.utils import setup_logger, seed_everything
from src.components.features_engineering import PreprocessorPipeline
from lightning.pytorch.profilers import SimpleProfiler
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

logger = setup_logger()

class HybridLSTM_FFNN(L.LightningModule):
    def __init__(self, input_size, lstm_hidden_size, ffnn_hidden_size, pos_weight, learning_rate=0.001):
        super().__init__()
        # Save only serializable hyperparameters (not tensors)
        self.save_hyperparameters(ignore=["pos_weight"])
        
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float)

        # === STREAM 1: The Feed-Forward Expert (from your successful model) ===
        self.ffnn_stream = nn.Sequential(
            nn.Linear(self.hparams.input_size, self.hparams.ffnn_hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.hparams.ffnn_hidden_size),
            nn.Dropout(p=0.4),
            nn.Linear(self.hparams.ffnn_hidden_size, self.hparams.ffnn_hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hparams.ffnn_hidden_size // 2),
        )
        
        # === STREAM 2: The LSTM Historian ===
        self.lstm_stream = nn.LSTM(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.lstm_hidden_size,
            num_layers=1, # Start simple
            batch_first=True,
            dropout=0.2
        )
        
        # === FINAL COMBINER ===
        ffnn_output_size = self.hparams.ffnn_hidden_size // 2
        combiner_input_size = ffnn_output_size + self.hparams.lstm_hidden_size
        
        self.combiner = nn.Sequential(
            # This is where the non-linear interaction happens
            nn.Linear(combiner_input_size, combiner_input_size // 2),
            nn.ReLU(), # The requested ReLU
            nn.BatchNorm1d(combiner_input_size // 2),
            nn.Dropout(p=0.2),
            nn.Linear(combiner_input_size // 2, 1)
        )
        self.ffnn_head = nn.Linear(ffnn_output_size, 1)
        self.lstm_head = nn.Linear(self.hparams.lstm_hidden_size, 1)
        # === METRICS ===
        self.train_ap = AveragePrecision(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.train_recall = Recall(task="binary")
        self.train_precision = Precision(task="binary")

        self.val_ap = AveragePrecision(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_precision = Precision(task="binary")

    def forward(self, x):
        # x is now a tuple: (historical_sequence, target_transaction_features)
        historical_x, target_x = x
        
        # --- Process Stream 1: FFNN Expert ---
        target_embedding = self.ffnn_stream(target_x)
        
        # --- Process Stream 2: LSTM Historian ---
        # We only need the final hidden state of the LSTM
        _, (h_n, _) = self.lstm_stream(historical_x)
        # h_n shape is [num_layers, batch, hidden_size], so we take the last layer's state
        context_embedding = h_n[-1]
        
        # --- Combine the expert opinions ---
        combined_embedding = torch.cat([target_embedding, context_embedding], dim=1)
        
        # --- Final Prediction through the combiner ---
        logits = self.combiner(combined_embedding)
        if not self.training:
            ffnn_logits = self.ffnn_head(target_embedding)
            lstm_logits = self.lstm_head(context_embedding)
            return logits, ffnn_logits, lstm_logits
        return logits
    def training_step(self, batch, batch_idx):
        (historical_x, target_x), y = batch
        
        # In training, `self.training` is True, so `forward` returns a SINGLE TENSOR.
        final_logits = self((historical_x, target_x))
        logits = final_logits.squeeze(1)
        
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.pos_weight)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        
        # Calculate training metrics
        train_auroc = self.train_auroc(probs, y)
        train_precision = self.train_precision(preds, y)
        train_recall = self.train_recall(preds, y)
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-8)
        
        self.log_dict({
            'train_loss': loss, 'train_auroc': train_auroc,
            'train_f1': train_f1, 'train_recall': train_recall
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        (historical_x, target_x), y = batch

        # In validation, `self.training` is False, so `forward` returns a TUPLE.
        # We must unpack it correctly.
        final_logits, _, _ = self((historical_x, target_x))
        logits = final_logits.squeeze(1)

        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.pos_weight)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        
        # Calculate validation metrics
        val_auroc = self.val_auroc(probs, y)
        val_precision = self.val_precision(preds, y)
        val_recall = self.val_recall(preds, y)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)

        self.log_dict({
            'val_loss': loss, 'val_auroc': val_auroc,
            'val_f1': val_f1, 'val_recall': val_recall
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auroc"}
        }
    
if __name__ == "__main__":
    batch_size = 512
    epoch = 20
    patience = 5
    pos_mul = 1.5
    hidden_size = 128
    train_duration=35
    test_duration=7
    seq_len = 21
    fast_dev_run=False
    seed_everything(42)
    
    factory = DataIngestorFactory()
    ingestor = factory.create_ingestor("duration_pkl")
    train_df, validation_df = ingestor.ingest(
        dir_path=rf"C:\Users\thuhi\workspace\fraud_detection\data\transformed_data",
        start_train_date="2018-05-15",
        train_duration=train_duration,
        test_duration=test_duration,
        delay=7
    )

    train_preprocessed = PreprocessorPipeline(train_df,add_method=["scale"]).process()
    validation_preprocessed = PreprocessorPipeline(validation_df,add_method=['scale']).process()
    pos_weight = pos_mul * 1/torch.tensor(train_preprocessed[DataIngestorConfig().output_feature].sum()
                              / (len(train_preprocessed) - train_preprocessed[DataIngestorConfig().output_feature].sum()))

    pos = train_df[DataIngestorConfig().output_feature].sum()
    pos_processed = train_preprocessed[DataIngestorConfig().output_feature].sum()

    # Preparing data
    train_data = BaggingSequentialFraudDetectionDataset(train_preprocessed, seq_len= seq_len)
    validation_data = BaggingSequentialFraudDetectionDataset(validation_preprocessed,seq_len =seq_len)
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory= True
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        shuffle=False,
        pin_memory= True

    )
    
    logger.info(f"total pos: {pos}, total pos after process: {pos_processed}")
    logger.info(f"total data: {len(train_df)}, total data after process: {len(train_preprocessed)}")
    logger.info(pos_weight)
    logger.info(len(train_data.config.input_features_transformed) * seq_len)

    #initiating
    run = 1
    log_dir = os.path.join("log", "LSTM_A_log")
    os.makedirs(log_dir, exist_ok=True)
    while True:
        version = f"run_{run}"
        if version in os.listdir(log_dir):
            run += 1
        else:
            logger = TensorBoardLogger(save_dir="log/", name="LSTM_A_log", version=version)
            profiler = SimpleProfiler(
                        dirpath=log_dir,
                        filename=f"A_Profiler_run{run}",
                        \
                        )
            break

    model = HybridLSTM_FFNN(
        input_size=len(train_data.config.input_features_transformed),
        lstm_hidden_size=128,  # Hyperparameter to tune
        ffnn_hidden_size=512,  # From your successful FFNN
        pos_weight=pos_weight,
        learning_rate=0.001
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True
    )

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=(
            f"bagging-{date_str}-run{run}"
            f"bs{batch_size}-ep{epoch}-pat{patience}-{pos_weight}-"
            f"traindur{train_duration}-testdur{test_duration}-hidden_size{hidden_size}"
            "{epoch:02d}-{val_loss:.2f}"
        ),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True
    )

    trainer = L.Trainer(
        fast_dev_run=fast_dev_run,
        num_sanity_val_steps=10,
        max_epochs=epoch,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[early_stop_callback,checkpoint_callback],
        profiler= profiler,
        accumulate_grad_batches=4
        
    )


    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )