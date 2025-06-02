import torch
import numpy as np



class EarlyStopping:
    def __init__(self, args):
        self.patience = args.early_stop
        self.min_delta = args.min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        
    
    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        elif loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True