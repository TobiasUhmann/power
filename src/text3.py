import os

from power.classifier.power_classifier import PowerClassifier


def main():
    if not os.path.isdir('data/'):
        os.mkdir('data/')

    power_classifier = PowerClassifier()


if __name__ == '__main__':
    main()
