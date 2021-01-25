from pytorch_lightning import Trainer

from ower.simple_classifier import SimpleClassifier
from ower.simple_data_module import SimpleDataModule


def main():
    data_module = SimpleDataModule(data_dir='data/', batch_size=64)

    classifier = SimpleClassifier(vocab_size=100000,
                                  embed_dim=32,
                                  num_class=4)

    trainer = Trainer(gpus=1)
    trainer.fit(classifier, data_module)


if __name__ == '__main__':
    main()
