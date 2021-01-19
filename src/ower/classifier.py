from argparse import Namespace

import pytorch_lightning as pl


class Classifier(pl.LightningModule):

    def __init__(self, hparams: Namespace):
        super().__init__()

        self.hparams = hparams
        self.batch_size = hparams.batch_size
