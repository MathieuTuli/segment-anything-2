from argparse import ArgumentParser

import torch

from sam2 import build_sam

parser = ArgumentParser(description="A CLI Starter Script")
parser.add_argument("--config", type=str, required="true")
parser.add_argument("--ckpt", type=str, required="true")


def convert(args):
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    convert(args)
