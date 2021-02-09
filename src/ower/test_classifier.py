import logging
from unittest import TestCase

import torch
from torch import tensor
from torch.nn import BCEWithLogitsLoss

from ower.classifier import Classifier


class TestClassifier(TestCase):
    def test_forward(self):
        #
        # GIVEN a classifier
        #

        classifier = Classifier(vocab_size=4, emb_size=4, class_count=3).cuda()

        classifier.class_embs = tensor([[1., 0., 0., 0.],
                                        [0., 1., 0., 0.],
                                        [0., 0., 1., 0.]]).cuda()

        classifier.embedding.weight.data = tensor([[1., 0., 0., 0.],
                                                   [0., 1., 0., 0.],
                                                   [0., 0., 1., 0.],
                                                   [0., 0., 0., 1.]]).cuda()

        classifier.fc.weight.data = tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]]).cuda()

        classifier.fc.bias.data = tensor([0., 0., 0.]).cuda()

        inputs_batch = tensor([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                               [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]]).cuda()

        outputs_batch = classifier(inputs_batch)
        print(outputs_batch)

    def test_forward_2(self):
        #
        # GIVEN a classifier
        #

        classifier = Classifier(vocab_size=4, emb_size=4, class_count=3).cuda()

        classifier.class_embs0 = tensor([[1., 0., 0., 0.],
                                         [0., 1., 0., 0.],
                                         [0., 0., 1., 0.]], requires_grad=True)

        classifier.class_embs = classifier.class_embs0.cuda()

        classifier.embedding.weight.data = tensor([[1., 0., 0., 0.],
                                                   [0., 1., 0., 0.],
                                                   [0., 0., 1., 0.],
                                                   [0., 0., 0., 1.]]).cuda()

        classifier.fc.weight.data = tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]]).cuda()

        classifier.fc.bias.data = tensor([0., 0., 0.]).cuda()

        labels_batch = tensor([[1., 1., 0.]]).cuda()
        optimizer = torch.optim.SGD(classifier.parameters(), lr=4.0)

        for i in range(10):
            print()
            print(i)

            inputs_batch = tensor([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]]).cuda()

            outputs_batch = classifier(inputs_batch)
            print(outputs_batch)

            loss = torch.nn.BCEWithLogitsLoss()(outputs_batch, labels_batch)
            print('### LOSS', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(classifier.class_embs)
            print(classifier.class_embs0)

    def test_forward_3(self):
        #
        # GIVEN a classifier
        #

        classifier = Classifier(vocab_size=4, emb_size=4, class_count=3).cuda()

        classifier.class_embs0 = tensor([[1., 0., 0., 0.],
                                         [0., 1., 0., 0.],
                                         [0., 0., 1., 0.]], requires_grad=True)

        classifier.class_embs = classifier.class_embs0.cuda()

        # classifier.embedding.weight.data = tensor([[0., 0., 0., 0.],
        #                                            [0., 0., 0., 0.],
        #                                            [0., 0., 0., 0.],
        #                                            [0., 0., 0., 0.]]).cuda()

        classifier.embedding.weight.data = torch.randn((4, 4)).cuda()

        classifier.fc.weight.data = tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]]).cuda()

        classifier.fc.bias.data = tensor([0., 0., 0.]).cuda()

        labels_batch = tensor([[1., 1., 0.]]).cuda()
        optimizer = torch.optim.SGD(classifier.parameters(), lr=4.0)

        for i in range(10):
            print()
            print(i)

            inputs_batch = tensor([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]]).cuda()

            outputs_batch = classifier(inputs_batch)
            print(outputs_batch)

            loss = torch.nn.BCEWithLogitsLoss()(outputs_batch, labels_batch)
            print('### LOSS', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(classifier.class_embs)
            print(classifier.class_embs0)

    def test_forward_4(self):
        logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.DEBUG)

        #
        # GIVEN a classifier
        #

        classifier = Classifier(vocab_size=4, emb_size=4, class_count=3).cuda()

        labels_batch = tensor([[1., 1., 0.]]).cuda()
        optimizer = torch.optim.SGD(classifier.parameters(), lr=4.0)

        for i in range(5):
            print()
            print(i)

            inputs_batch = tensor([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]]).cuda()

            outputs_batch = classifier(inputs_batch)
            print(outputs_batch)

            loss = torch.nn.BCEWithLogitsLoss()(outputs_batch, labels_batch)
            print('### LOSS', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('classifier.embedding.weight', classifier.embedding_bag.weight)
            print('classifier.class_embs', classifier.class_embs)
