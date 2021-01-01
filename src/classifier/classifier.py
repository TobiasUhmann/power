from pytorch_lightning import LightningModule


class Classifier(LightningModule):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self,):
        pass

    def forward(self, batch):
        pass

    def training_step(self, batch, _):
        pass
