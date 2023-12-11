import torch
import pytorch_lightning as pl
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE=4

class data_module(pl.LightningDataModule):

    def __init__(self):
        super(data_module,self).__init__()
  
    def setup(self,stage=None):
        if stage=='fit':
            X=np.loadtxt("X_preprocessed.csv", dtype=np.float32, delimiter=',')
            y=np.loadtxt("y.csv", dtype=np.float32, delimiter=',')
            indices=np.arange(X.shape[0])
            train_indices=np.random.choice(indices,int(0.8*X.shape[0]))
            indices=list(set(indices).difference(set(train_indices)))
            val_indices=np.random.choice(indices,int(0.1*X.shape[0]))
            test_indices=list(set(indices).difference(set(val_indices)))

            self.X_train,self.y_train=X[train_indices],y[train_indices]
            self.X_val,self.y_val=X[val_indices],y[val_indices]
            self.X_test,self.y_test=X[test_indices],y[test_indices]

    def train_dataloader(self):        
        return DataLoader(TensorDataset(torch.from_numpy(self.X_train),torch.from_numpy(self.y_train)), batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.from_numpy(self.X_val),torch.from_numpy(self.y_val)), batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(TensorDataset(torch.from_numpy(self.X_test),torch.from_numpy(self.y_test)), batch_size=BATCH_SIZE, shuffle=False, num_workers=8)    