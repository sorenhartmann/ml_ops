import pytorch_lightning as pl
from argparse import ArgumentParser
from src.models.model import MyAwesomeModel
from src.data.make_dataset import MNISTDataModule
import torch
from azureml.core import Model, Workspace
from pathlib import Path

def main(args):

    wandb_logger = pl.loggers.WandbLogger(project="mnist")
    dm = MNISTDataModule("./data/")
    model = MyAwesomeModel(**vars(args))
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        progress_bar_refresh_rate=0,
        callbacks=[pl.callbacks.EarlyStopping("val_loss", patience=5)],
    )
    trainer.fit(model, dm)

    if args.register_with_azure:
        ws = Workspace.from_config(".azure")
        model_path = Path("model_state_dict.pt")
        torch.save(model.state_dict(), model_path)
        Model.register(ws, model_path, "mnist") 
        model_path.unlink()

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--register_with_azure", action="store_true")

    parser = MyAwesomeModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
