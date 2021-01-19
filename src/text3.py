from pytorch_lightning import Trainer

from ower.old_classifier import OldClassifier
from ower.old_data_module import OldDataModule


def main():
    data_module = OldDataModule(data_dir='data/', batch_size=64)

    classifier = OldClassifier(vocab_size=100000,
                               embed_dim=32,
                               num_class=4)

    trainer = Trainer(gpus=1)
    trainer.fit(classifier, data_module)


if __name__ == '__main__':
    main()
