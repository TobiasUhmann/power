#
# PyTorch Lightning version of torchtext_classifier_refactored.py
#

import pickle
from typing import List

import torch
import torch.nn as nn
import torchtext
from pytorch_lightning import Trainer
from torch import Tensor
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import Vocab

from snippets.torchtext_classifier_lightning.classifier import Classifier
from snippets.torchtext_classifier_lightning.data_module import DataModule

BATCH_SIZE = 16
EMBED_DIM = 32
NGRAMS = 2
NUM_EPOCHS = 5


def main():
    #
    # Load data and instantiate classifier
    #

    data_module = DataModule(data_dir='data/', batch_size=BATCH_SIZE, ngrams=NGRAMS)
    data_module.prepare_data()

    with open('data/data_module.pkl', 'wb') as f:
        pickle.dump(data_module, f)

    with open('data/data_module.pkl', 'rb') as f:
        data_module = pickle.load(f)

    vocab_size = len(data_module.vocab)
    num_classes = data_module.num_classes

    classifier = Classifier(vocab_size, EMBED_DIM, num_classes)

    #
    # Train & Test
    #

    trainer = Trainer(max_epochs=NUM_EPOCHS, gpus=1)
    trainer.fit(classifier, datamodule=data_module)

    trainer.save_checkpoint('data/classifier.ckpt')

    trainer.test(classifier, datamodule=data_module)

    #
    # Predict custom sample
    #

    class_to_label = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tec'}

    def predict(text: str, classifier: Classifier, vocab: Vocab, ngrams: int):
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        words: List[str] = tokenizer(text)

        with torch.no_grad():
            tokens = torch.tensor([vocab[ngram] for ngram in ngrams_iterator(words, ngrams)])

            class_logits: Tensor = classifier(tokens, torch.tensor([0]))
            pred_class = class_logits.argmax(1).item() + 1

            return pred_class

    ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was enduring the season’s" \
                  " worst weather conditions on Sunday at The Open on his way to a" \
                  " closing 75 at Royal Portrush, which considering the wind and the" \
                  " rain was a respectable showing. Thursday’s first round at the" \
                  " WGC-FedEx St. Jude Invitational was another story. With temperatures" \
                  " in the mid-80s and hardly any wind, the Spaniard was 13 strokes" \
                  " better in a flawless round. Thanks to his best putting performance" \
                  " on the PGA Tour, Rahm finished with an 8-under 62 for a" \
                  " three-stroke lead, which was even more impressive considering he’d" \
                  " never played the front nine at TPC Southwind."

    classifier = classifier.to('cpu')
    vocab = data_module.vocab

    pred_class = predict(ex_text_str, classifier, vocab, ngrams=2)
    pred_label = class_to_label[pred_class]

    print()
    print(f'This is a {pred_label} news')


if __name__ == '__main__':
    main()
