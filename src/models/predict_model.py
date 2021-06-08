from pathlib import Path

import click
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from src.data.make_dataset import transform

@click.command()
@click.argument("model_file", type=click.Path(exists=True))
@click.argument("image_path", type=click.Path(exists=True))
def main(model_file, image_path):

    image_path = Path(image_path)

    print("Predicting...")

    model = torch.load(model_file)
    model.eval()

    if image_path.is_dir():
        images = [f for f in image_path.iterdir() if f.suffix == ".png"]
    elif image_path.suffix == ".png":
        images = [image_path]
    else:
        raise ValueError()

    images = sorted(images)    

    for image_file in images:
        with Image.open(image_file) as f:
            t = ToTensor()(f)
        with torch.autograd.no_grad():
            t = t.to(torch.float)
            t = transform(t)
            proba = model(t.unsqueeze(0)).softmax(-1)
            print(f"File: {image_file.name} -> {proba.argmax(-1).item()} ({100*proba.max():02.2f}%)")



if __name__ == "__main__":
    main()
