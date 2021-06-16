from kaggle.api.kaggle_api_extended import KaggleApi
import wandb
from src.models.model import MyAwesomeModel
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from src.data.make_dataset import MNISTDataModule
import torch

wandb_user = "sorenhartmann"
wandb_project = "mnist"

def get_test_data():

    test_data = pd.read_csv("data/raw/test.csv")
    test_data = torch.tensor(test_data.to_numpy()).to(dtype=torch.float)
    test_data = test_data.reshape(-1, 1, 28, 28)
    return test_data

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)

    args = parser.parse_args()

    # Get runs from wandb
    wandb_api = wandb.Api()
    runs = wandb_api.runs(f"{wandb_user}/{wandb_project}")

    # Get relevant run
    if args.model_name is None:
        run = next(run for run in runs if run.state == "finished")
    else:
        run = next(run for run in runs if run.name == args.model_name)

    # Find checkpoint file, possibly downloading into ./models/
    checkpoint_file = next(file for file in run.files() if file.name.endswith(".ckpt"))

    try:
        local_path = next(
            Path("./wandb").glob(f"*-{run.id}/files/{checkpoint_file.name}")
        )
    except StopIteration:
        local_path = Path("models") / checkpoint_file.name
        if not local_path.exists():
            checkpoint_file.download("models").close()

    model = MyAwesomeModel.load_from_checkpoint(local_path)

    test_data = get_test_data()
    output = model(test_data)

    result_df = pd.DataFrame(
        {
            "ImageId": torch.arange(1, len(output) + 1),
            "Label": output.argmax(-1),
        }
    )

    result_df.to_csv("submission.csv", index=False)

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    kaggle_api.competition_submit(
        "submission.csv", message=f"Test submission run={run.name}", competition="digit-recognizer"
    )