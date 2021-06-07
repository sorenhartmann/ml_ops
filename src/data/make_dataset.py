# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms

data_dir = Path(__file__).parents[2] / "data"

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

class MNIST(datasets.MNIST):
    
    @property
    def raw_folder(self) -> str:
        return self.root / 'raw' / "MNIST"

    @property
    def processed_folder(self) -> str:
        return self.root / 'processed' / "MNIST"

def mnist():
    

    # Download and load the training data
    train = MNIST(data_dir, download=True, train=True, transform=transform)
    # Download and load the test data
    test = MNIST(data_dir, download=True, train=False, transform=transform)

    return train, test

def main():

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    logger.info("downloading MNIST dataset")
    
    mnist()

    logger.info("done")

if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
