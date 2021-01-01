import os

from classifier.classifier import Classifier


def main():
    if not os.path.isdir('data/'):
        os.mkdir('data/')

    classifier = Classifier()


if __name__ == '__main__':
    main()
