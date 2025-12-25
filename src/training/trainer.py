from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.utils import load_checkpoint, save_training_history
from src.training.callbacks import CheckpointManager, EarlyStopping, EarlyStoppingConfig
from src.training.metrics import accuracy


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict,
        scheduler=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.mixed_precision = config["training"].get("mixed_precision", False)
        # Enable mixed precision scaling only when running on CUDA devices.
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.mixed_precision and torch.cuda.is_available()
        )

        logging_cfg = config["logging"]
        Path(logging_cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
        Path(logging_cfg["metrics_dir"]).mkdir(parents=True, exist_ok=True)
        Path(logging_cfg["plots_dir"]).mkdir(parents=True, exist_ok=True)
        Path(logging_cfg["tensorboard_dir"]).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(logging_cfg["tensorboard_dir"])

        early_cfg = config.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            EarlyStoppingConfig(
                patience=early_cfg.get("patience", 10),
                min_delta=early_cfg.get("min_delta", 0.0),
                monitor=early_cfg.get("monitor", "val_loss"),
            )
        )
        self.checkpoint_manager = CheckpointManager(config["model"]["checkpoint_dir"])
        self.best_checkpoint_path: Optional[str] = None
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def fit(self) -> Dict[str, list]:
        num_epochs = self.config["training"]["num_epochs"]
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate(epoch)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            self.writer.add_scalars(
                "Loss",
                {"train": train_loss, "val": val_loss},
                epoch,
            )
            self.writer.add_scalars(
                "Accuracy",
                {"train": train_acc, "val": val_acc},
                epoch,
            )

            ckpt_path = self.checkpoint_manager.save_best(
                self.model, self.optimizer, epoch, val_loss, self.history
            )
            if ckpt_path:
                self.best_checkpoint_path = ckpt_path
                print(f"Saved new best model to {ckpt_path}")

            if self.scheduler:
                self.scheduler.step()

            self.early_stopping.step(val_loss)
            if self.early_stopping.should_stop:
                print("Early stopping triggered.")
                break

        if self.best_checkpoint_path:
            load_checkpoint(self.model, self.best_checkpoint_path, self.device)

        metrics_path = Path(self.config["logging"]["metrics_dir"]) / "training_history.json"
        save_training_history(self.history, str(metrics_path))
        self.writer.close()
        return self.history

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for step, batch in enumerate(progress, start=1):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=self.mixed_precision and torch.cuda.is_available()
            ):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            gradient_clip = self.config["training"].get("gradient_clip")
            if gradient_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_acc = accuracy(outputs.detach(), labels)
            running_loss += loss.item()
            running_acc += batch_acc

            progress.set_postfix({"loss": running_loss / step, "acc": running_acc / step})

        steps = len(self.train_loader)
        return running_loss / steps, running_acc / steps

    def _validate(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0

        with torch.no_grad():
            progress = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            for step, batch in enumerate(progress, start=1):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                running_acc += accuracy(outputs, labels)

                progress.set_postfix({"loss": running_loss / step, "acc": running_acc / step})

        steps = len(self.val_loader)
        return running_loss / steps, running_acc / steps

