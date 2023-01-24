import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class Trainer():

    def __init__(self, config, model, optimizer, criterion, device):

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.model.to(self.device)

    def _train_step(self, dataloader, epoch):
        
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        for batch_idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch + 1}")

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
            
            if ((val_dataloader is not None) and (((epoch + 1) % self.config.training.evaluate_every)) == 0):
                val_loss = self.evaluate(val_dataloader)
                
                # if self.best_val_loss >= val_loss and self.config.save_model_optimizer:
                #   self.best_val_loss = val_loss
                #   print("Saving best model and optimizer at checkpoints/{}/model_optimizer.pt".format(self.config.load_path))
                #   os.makedirs("checkpoints/{}/".format(self.config.load_path), exist_ok=True)
                #   torch.save({
                #         'model_state_dict': self.model.state_dict(),
                #         'optimizer_state_dict': self.optimizer.state_dict(),
                #       }, "checkpoints/{}/model_optimizer.pt".format(self.config.load_path))
                self.model.train()

    def evaluate(self, dataloader):
        self.model.eval()
        
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        tepoch.set_description("Validation Step")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tepoch):
                    
                out = self.model(batch["sample"].to(self.device))
                loss = self.criterion(out, batch["target"].to(self.device))

                total_loss += loss.item()
                tepoch.set_postfix(loss = total_loss / (batch_idx+1))

        return (total_loss / (batch_idx + 1))

    def compute_metrics(self, dataloader):

        batch_iou = []
        self.model.eval()
        
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        tepoch.set_description("Computing metrics")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tepoch):
                    
                out = self.model(batch["sample"].to(self.device)).detach().cpu()
                out = torch.argmax(out, dim = 1)

                batch_iou.extend(self._iou(out, batch["target"], eps = self.config.metrics.eps).detach().cpu().tolist())

        # iou = self._iou(pr, gt, eps = self.config.metrics.eps, threshold = self.config.metrics.threshold, ignore_channels = self.config.metrics.ignore_channels)
        # iou = self._iou(pr, gt, eps = self.config.metrics.eps)
        iou = np.mean(batch_iou)
        return iou


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


    # def _take_channels(self, *xs, ignore_channels = None):
    #     if ignore_channels is None:
    #         return xs
    #     else:
    #         channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
    #         xs = [torch.index_select(x, dim = 1, index = torch.tensor(channels).to(x.device)) for x in xs]
    #         return 

    # # Function to modify x based on the threshold
    # def _threshold(self, x, threshold = None):
    #     if threshold is not None:
    #         return (x > threshold).type(x.dtype)
    #     else:
    #         return x

    # # Function to calculate the IoU score
    # def _iou(self, pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    #     """Calculate Intersection over Union between ground truth and prediction
    #     Args:
    #         pr (torch.Tensor): predicted tensor
    #         gt (torch.Tensor):  ground truth tensor
    #         eps (float): epsilon to avoid zero division
    #         threshold: threshold for outputs binarization
    #     Returns:
    #         float: IoU (Jaccard) score
    #     """

    #     pr = self._threshold(pr, threshold = threshold)
    #     pr, gt = self._take_channels(pr, gt, ignore_channels = ignore_channels)

    #     intersection = torch.sum(gt * pr)
    #     union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    #     return (intersection + eps) / union