import torch
import torch.nn as nn

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
                
                if self.best_val_loss >= val_loss and self.config.save_model_optimizer:
                  self.best_val_loss = val_loss
                  print("Saving best model and optimizer at checkpoints/{}/model_optimizer.pt".format(self.config.load_path))
                  os.makedirs("checkpoints/{}/".format(self.config.load_path), exist_ok=True)
                  torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                      }, "checkpoints/{}/model_optimizer.pt".format(self.config.load_path))
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

        pr = []
        gt = []
        self.model.eval()
        
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        tepoch.set_description("Computing metrics")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tepoch):
                    
                out = self.model(batch["sample"].to(self.device))
                pr.extend(out.detach().cpu().to_list())
                gt.extend(batch["target"])

        iou = self._iou(pr, gt, eps = self.config.metrics.eps, threshold = self.config.metrics.threshold, ignore_channels = self.config.metrics.ignore_channels)

        return iou


    def _take_channels(self, *xs, ignore_channels = None):
        if ignore_channels is None:
            return xs
        else:
            channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
            xs = [torch.index_select(x, dim = 1, index = torch.tensor(channels).to(x.device)) for x in xs]
            return 

    # Function to modify x based on the threshold
    def _threshold(self, x, threshold = None):
        if threshold is not None:
            return (x > threshold).type(x.dtype)
        else:
            return x

    # Function to calculate the IoU score
    def _iou(self, pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
        """Calculate Intersection over Union between ground truth and prediction
        Args:
            pr (torch.Tensor): predicted tensor
            gt (torch.Tensor):  ground truth tensor
            eps (float): epsilon to avoid zero division
            threshold: threshold for outputs binarization
        Returns:
            float: IoU (Jaccard) score
        """

        pr = _threshold(pr, threshold = config.evaluate.threshold)
        pr, gt = _take_channels(pr, gt, ignore_channels = config.evaluate.ignore_channels)

        intersection = torch.sum(gt * pr)
        union = torch.sum(gt) + torch.sum(pr) - intersection + eps
        return (intersection + eps) / union