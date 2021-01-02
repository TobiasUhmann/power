from pytorch_lightning import Trainer

from power.classifier import Classifier
from power.data_module import DataModule


def main():
    data_module = DataModule(data_dir='data/', batch_size=64)

    classifier = Classifier(vocab_size=100000,
                            embed_dim=32,
                            num_class=4)

    trainer = Trainer()
    trainer.fit(classifier, data_module)


if __name__ == '__main__':
    main()
