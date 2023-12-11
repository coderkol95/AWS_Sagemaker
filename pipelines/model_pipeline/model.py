import pytorch_lightning as pl
import torch
from torch.nn.functional import relu, mse_loss
from torch.optim import Adam
from datawork import data_module
import argparse
import os
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Pytorch 2 has problem with last linear layer having 1 cell. Hence reverted to prev version

EPOCHS=3


# Important path for sagemaker
prefix = '/' # opt/ml/

input_path = os.path.join(prefix, 'input/')
output_path = os.path.join(prefix, 'output/')
model_path = os.path.join(prefix, 'model/')

if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Defining training channel
# channel_name = 'training'
# training_path = os.path.join(input_path, channel_name)

class NN(pl.LightningModule):

    def __init__(
            self, 
            dropout:float,
            output_dims: List[int],
            lr:float) -> None:

        super().__init__()
        self.lr=lr
        self.loss=mse_loss

        layers: List[nn.Module] = []

        input_dim: int = 5
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, 1)) # Here would have been CLASSES

        self.layers = nn.Sequential(*layers)

    def configure_optimizers(self):
        return Adam(params=self.parameters(), lr=self.lr)

    def forward(self, data):
        
        # If logits were returned, you would have returned the F.softmax etc. 
        return self.layers(data)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        preds = self.forward(x)
        loss = self.loss(preds.flatten(), y) 
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, valid_batch, batch_idx): 
        x, y = valid_batch 
        preds = self.forward(x) 
        loss = self.loss(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, test_batch, batch_idx): 
        x, y = test_batch 
        logits = self.forward(x) 
        loss = self.loss(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

def objective(trial):

    # We optimize the number of layers, hidden units in each layer, dropout and the learning rate.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    lr = trial.suggest_float("learning_rate",1e-5,1e-1)

    output_dims = [
        trial.suggest_int(f"n_units_l{i}", 4, 128, log=True) for i in range(n_layers)
    ]

    model = NN(dropout, output_dims,lr)
    data=data_module()

    trainer = pl.Trainer(
        logger=True,
        # limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        default_root_dir=output_path
    )
    hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims, lr=lr)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data)

    return trainer.callback_metrics["val_loss"].item()

if __name__ =='__main__':

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    epochs=2

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=10, timeout=600)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Only running experiments in this docker container.
    # Also need to save the hyperparams and loss metrics per epoch in this run

    # sys.exit(0)
 