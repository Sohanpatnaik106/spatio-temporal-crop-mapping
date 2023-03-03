import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

class Trainer():

    def __init__(self, config, model, optimizer, criterion, device):

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        if self.config.sched_function == "cosine_annealing_lr":
            self.scheduler = CosineAnnealingLR(self.optimizer, self.config.training.epochs)
        elif self.config.sched_function == "multi_step_lr":
            self.scheduler = MultiStepLR(self.optimizer, [i * 5 for i in range(self.config.training.epochs)])

        self.best_val_loss = np.finfo(np.float32).max

        self.model.to(self.device)

    def _train_step(self, dataloader, epoch):
        
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        for batch_idx, batch in enumerate(tepoch):
            tepoch.set_description(f"INFO: Epoch {epoch + 1}")

            out = self.model(batch["sample"].to(self.device))

            loss = self.criterion(out, batch["target"].to(self.device))
            loss.backward()

            total_loss += loss.item()
            tepoch.set_postfix(loss = total_loss / (batch_idx + 1))
            
            self.optimizer.step()
            self.optimizer.zero_grad()

        return (total_loss / (batch_idx + 1))
    
    def train(self, train_dataloader, val_dataloader=None):
        self.model.train()
        for epoch in range(self.config.training.epochs):
            self._train_step(train_dataloader, epoch)
            self.scheduler.step()
            
            if ((val_dataloader is not None) and (((epoch + 1) % self.config.training.evaluate_every)) == 0):
                val_loss = self.evaluate(val_dataloader)
                
                if self.best_val_loss >= val_loss and self.config.save_model_optimizer:
                  self.best_val_loss = val_loss
                  print(f"Saving best model and optimizer at checkpoints/{self.config.model.model_name}/model_optimizer.pt")
                  os.makedirs(f"checkpoints/{self.config.model.model_name}/", exist_ok = True)
                  torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                      }, f"checkpoints/{self.config.model.model_name}/model_optimizer.pt")
                self.model.train()

    def evaluate(self, dataloader):
        self.model.eval()
        
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        tepoch.set_description("INFO: Validation Step")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tepoch):
                    
                out = self.model(batch["sample"].to(self.device))
                loss = self.criterion(out, batch["target"].to(self.device))

                total_loss += loss.item()
                tepoch.set_postfix(loss = total_loss / (batch_idx+1))

        return (total_loss / (batch_idx + 1))

    def compute_metrics(self, dataloader):

        batch_iou = []
        batch_accuracy = []
        batch_precision = []
        batch_recall = []
        batch_f1_score = []
        self.model.eval()
        
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        tepoch.set_description("Computing metrics")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tepoch):
                    
                out = self.model(batch["sample"].to(self.device)).detach().cpu()
                out = torch.argmax(out, dim = 1)

                batch_iou.extend(self._iou(out, batch["target"], eps = self.config.metrics.eps).detach().cpu().tolist())
                batch_accuracy.append(self._accuracy(out, batch["target"]))
                batch_precision.append(self._precision(out, batch["target"]))
                batch_recall.append(self._recall(out, batch["target"]))
                batch_f1_score.append(self._f1_score(out, batch["target"]))

        results = {
            "accuracy": np.mean(batch_accuracy),
            "precision": np.mean(batch_precision),
            "recall": np.mean(batch_recall),
            "f1_score": np.mean(batch_f1_score),
            "iou": np.mean(batch_iou)
        }

        return results


    def _iou(self, outputs: torch.Tensor, labels: torch.Tensor, eps: float):
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape
        outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

        iou = (intersection + eps) / (union + eps)  # We smooth our devision to avoid 0/0

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

        return thresholded  # Or thresholded.mean() if you are interested in average across the batch

    def _accuracy(self, outputs, labels):
        # NOTE: Weighted accuracy can be computed later
        # TODO: Implement class wise accuracy
        outputs = outputs.numpy().flatten()
        labels = labels.numpy().flatten()
        # print(outputs, labels)
        return accuracy_score(outputs, labels)

    def _precision(self, outputs, labels):
        # NOTE: Weighted precision can be computed later
        # TODO: Implement class wise precision
        outputs = outputs.numpy().flatten()
        labels = labels.numpy().flatten()
        return precision_score(outputs, labels, average='micro')

    def _recall(self, outputs, labels):
        # NOTE: Weighted recall can be computed later
        # TODO: Implement class wise recall
        outputs = outputs.numpy().flatten()
        labels = labels.numpy().flatten()
        return recall_score(outputs, labels, average='micro')

    def _f1_score(self, outputs, labels):
        # NOTE: Weighted f1_score can be computed later
        # TODO: Implement class wise f1_score
        outputs = outputs.numpy().flatten()
        labels = labels.numpy().flatten()
        return f1_score(outputs, labels, average='micro')