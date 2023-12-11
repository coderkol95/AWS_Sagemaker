import pytorch_lightning as pl
import torch
from torch.nn.functional import relu, mse_loss
from torch.optim import Adam
from datawork import data_module
import argparse
import os
import sys

# Important path for sagemaker
prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')

# Defining training channel
# channel_name = 'training'
# training_path = os.path.join(input_path, channel_name)

class NN(pl.LightningModule):

    def __init__(self, lr):
        super().__init__()
        self.lr=lr
        self.loss=mse_loss
        self.inp=torch.nn.Linear(in_features=5,out_features=3)
        # self.h1=torch.nn.Linear(in_features=10,out_features=10)
        self.out=torch.nn.Linear(in_features=3,out_features=1)

    def configure_optimizers(self):
        return Adam(params=self.parameters(), lr=self.lr)
      
    def forward(self, X):
        
        self.X = relu(self.inp(X))
        # self.X = relu(self.h1(self.X))
        self.X = relu(self.out(self.X))
        return relu(self.X)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        logits = self.forward(x)
        loss = self.loss(logits, y) 
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, valid_batch, batch_idx): 
        x, y = valid_batch 
        logits = self.forward(x) 
        loss = self.loss(logits, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, test_batch, batch_idx): 
        x, y = test_batch 
        logits = self.forward(x) 
        loss = self.loss(logits, y)
        self.log("test_loss", loss)
        return loss

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=5)

    args, _ = parser.parse_known_args()
    print(args)

    trainer=pl.Trainer(gpus=1, max_epochs=args.epochs, default_root_dir=args.output_data_dir)

    model = NN()
    data=data_module()

    print("Running model training.")
    # Runs model training
    trainer.fit(model,data)
    print("Model trained. check if its saved.")

    # After model has been trained, save its state into model_dir which is then copied to back S3
    with open(os.path.join(args.model_dir, os.path.join(model_path,f'model_{args.epochs}.pth')), 'wb') as f:
        torch.save(model.state_dict(), f)

    # Also need to save the hyperparams and loss metrics per epoch in this run

    sys.exit(0)
 