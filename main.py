from pytorch_lightning.cli import ArgsType, LightningCLI
from models.model_interface import MInterface
from datasets.data_interface import DInterface
import os


def cli_main():
    cli = LightningCLI(MInterface, DInterface)


if __name__ == '__main__':
    cli_main()

