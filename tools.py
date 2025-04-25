import random
import os
import math
import numpy as np
import torch

# set the random seeds
def random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, verbose=False, save_path='.', delta=0, patience=7):
        '''
        Args:
            verbose: If True, prints a message for each validation loss improvement.
                     Default: False
            save_path: Path for the checkpoint to be saved to.
                     Default: '.'
            delta: Minimum change in the monitored quantity to qualify as an improvement.
                     Default: 0
            patience: How long to wait after last time validation loss improved.
                     Default: 7
        '''
        self.best_score = None
        self.verbose = verbose
        self.save_path = save_path
        self.val_loss_min = np.Inf
        self.delta = delta
        self.counter = 0
        self.patience = patience
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, val_loss: {val_loss}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


def time_since(start, end):
    s = end - start
    m = math.floor(s/60)
    s -= m * 60
    return f'total time:{m}m {s}s'

def get_pearson_corr(y_true, y_pred):
    fsp = y_pred - torch.mean(y_pred)
    fst = y_true - torch.mean(y_true)
    devP = torch.std(y_pred)
    devT = torch.std(y_true)
    return torch.mean(fsp * fst) / devP * devT




