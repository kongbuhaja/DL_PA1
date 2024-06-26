# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
import torch

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting
from src.ViT import VisionTransformer
from src.AlexNet import AlexNet


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
                 metric: str = 'recall'
        ):
        super().__init__()
        self.num_classes = num_classes

        # Network
        if model_name == 'AlexNet':
            self.model = AlexNet(cfg)
        elif model_name == 'ViT':
            self.model = VisionTransformer(cfg)
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        if metric=='recall':
            self.metric = MyAccuracy()
            self.loging = lambda loss, performance, task: self.log_dict({f'loss/{task}': loss, f'{metric}/{task}': performance}, 
                                                                     on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.metric = MyF1Score(num_classes)
            self.loging = lambda loss, performance, task: self.log_dict({f'loss/{task}': loss, f'{metric}/{task}':performance[0], 
                                                                      f'precision/{task}': performance[1], f'recall/{task}': performance[2]}, 
                                                                     on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Hyperparameters
        self.save_hyperparameters()

    def one_hot(self, y, smooth=True, alpha=0.1):
        onehot = torch.zeros((y.shape[0], self.num_classes), dtype=torch.float32)
        b = torch.arange(y.shape[0])
        onehot[b, y] = 1.
        if smooth:
            onehot * (1 - alpha) + alpha/self.num_classes

        return onehot.to(y.device)

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        performance = self.metric(scores, y)
        self.loging(loss, performance, 'train')
  
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        performance = self.metric(scores, y)
        self.loging(loss, performance, 'val')
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        y_ = self.one_hot(y)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y_)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
