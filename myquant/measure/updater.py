import math
import torch

from myquant.measure.cosine import torch_cosine_similarity
from myquant.measure.norm import torch_mean_square_error
class BaseUpdater:
    def __init__(self):
        self.n = 0

    def update(self, y_pred, y_real):
        self.n += 1
    
    @property
    def data(self):
        return self.n

class MSEUpdater(BaseUpdater):
    def __init__(self):
        super().__init__()
        self.mse_sum = 0
    def update(self, y_pred:torch.Tensor, y_real:torch.Tensor):
        super().update(y_pred, y_real)
        # error = (y_pred - y_real).float().pow(2).mean().item()
        error = torch_mean_square_error(y_pred,y_real)
        self.mse_sum += error

    @property
    def data(self):
        mse = self.mse_sum / self.n
        return mse

class EQCosineUpdater(BaseUpdater):
    def __init__(self):
        super().__init__()
        self.cos_sum = 0
    def update(self, y_pred:torch.Tensor, y_real:torch.Tensor):
        super().update(y_pred, y_real)
        error = torch_cosine_similarity(y_pred,y_real)
        self.cos_sum += error

    @property
    def data(self):
        mse = self.cos_sum / self.n
        return -mse

class CosineUpdater(BaseUpdater):
    def __init__(self):
        super().__init__()
        self.dot_product_sum = 0
        self.y_pred_norm_squared = 0
        self.y_real_norm_squared = 0

    def update(self, y_pred: torch.Tensor, y_real: torch.Tensor):
        super().update(y_pred, y_real)
        y_pred = y_pred.float()
        y_real = y_real.float()
        
        dot_product = torch.sum(y_pred * y_real).item()
        y_pred_norm = torch.sum(y_pred.pow(2)).item()
        y_real_norm = torch.sum(y_real.pow(2)).item()

        self.dot_product_sum += dot_product
        self.y_pred_norm_squared += y_pred_norm
        self.y_real_norm_squared += y_real_norm

    @property
    def data(self):
        cosine_similarity = self.dot_product_sum / (torch.sqrt(torch.tensor(self.y_pred_norm_squared)) * 
                                                    torch.sqrt(torch.tensor(self.y_real_norm_squared)))
        # 这里作为loss取个负数，cosine_similarity越大越接近于1越相似
        # mse中越小越相似
        return -cosine_similarity.item()