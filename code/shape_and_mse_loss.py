import torch
import torch.nn as nn

class ShapeBasedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
       
        super(ShapeBasedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, pred, target):
        
        pred_norm = (pred - pred.min(dim=2, keepdim=True)[0]) / (pred.max(dim=2, keepdim=True)[0] - pred.min(dim=2, keepdim=True)[0])
        target_norm = (target - target.min(dim=2, keepdim=True)[0]) / (target.max(dim=2, keepdim=True)[0] - target.min(dim=2, keepdim=True)[0])
        
        shape_loss = torch.mean((pred_norm - target_norm) ** 2)
        
        pred_grad = pred_norm[:, :, 1:] - pred_norm[:, :, :-1]
        target_grad = target_norm[:, :, 1:] - target_norm[:, :, :-1]
        gradient_loss = torch.mean((pred_grad - target_grad) ** 2)
  
        pred_grad2 = pred_grad[:, :, 1:] - pred_grad[:, :, :-1]
        target_grad2 = target_grad[:, :, 1:] - target_grad[:, :, :-1]
        curvature_loss = torch.mean((pred_grad2 - target_grad2) ** 2)
        
        total_loss = self.alpha * shape_loss + self.beta * gradient_loss + self.gamma * curvature_loss
        
        return total_loss

class CombinedLoss(nn.Module):
    def __init__(self, shape_weight=0.5, mse_weight=0.5):
        
        super(CombinedLoss, self).__init__()
        self.shape_loss = ShapeBasedLoss()
        self.mse_loss = nn.MSELoss()
        self.shape_weight = shape_weight
        self.mse_weight = mse_weight
        
    def forward(self, pred, target):
        shape_loss = self.shape_loss(pred, target)
        mse_loss = self.mse_loss(pred, target)
        
        return self.shape_weight * shape_loss + self.mse_weight * mse_loss
        # return shape_loss
