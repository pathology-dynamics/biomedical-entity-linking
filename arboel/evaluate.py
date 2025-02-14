import sys

sys.path.append("../..")
import os

current_path = os.getcwd()
print(current_path)
from blink.common.params import BlinkParser

import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from blink.biencoder.LightningModule import LitArboel
from blink.biencoder.LightningDataModule import ArboelDataModule


def main(args):
    data_module = ArboelDataModule(params=args)

    MyModel = LitArboel.load_from_checkpoint(
        params=args, checkpoint_path=args["model_checkpoint"]
    )

    trainer = L.Trainer(
        limit_test_batches=0.01,
        devices=args["devices"],
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        precision="16-mixed",
    )

    trainer.test(model=MyModel, datamodule=data_module)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
