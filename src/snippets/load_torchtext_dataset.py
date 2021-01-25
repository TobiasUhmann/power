from torchtext.datasets import text_classification


def main():
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root='data/', ngrams=2, vocab=None)

    return


if __name__ == '__main__':
    main()
