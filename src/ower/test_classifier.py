from unittest import TestCase

from torch import tensor

from ower.classifier import Classifier


class TestClassifier(TestCase):
    def test_forward(self):

        #
        # GIVEN a classifier
        #

        classifier = Classifier(vocab_size=10, emb_size=4, num_classes=3).cuda()

        classifier.class_embs = tensor([[1., 0., 0., 0.],
                                        [0., 1., 0., 0.],
                                        [0., 0., 1., 0.]]).cuda()

        classifier.embedding.weight.data = tensor([[.0, .0, .0, .0],
                                                   [.0, .1, .0, .0],
                                                   [.0, .0, .2, .0],
                                                   [.0, .0, .0, .3],
                                                   [.4, .0, .0, .0],
                                                   [.0, .5, .0, .0],
                                                   [.0, .0, .6, .0],
                                                   [.0, .0, .0, .7],
                                                   [.8, .0, .0, .0],
                                                   [.0, .9, .0, .0]]).cuda()

        classifier.fc.weight.data = tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]).cuda()

        classifier.fc.bias.data = tensor([0., 0., 0.]).cuda()

        inputs_batch = tensor([[[1, 2, 3, 4, 5],
                                [2, 3, 4, 5, 6]],

                               [[3, 4, 5, 6, 7],
                                [4, 5, 6, 7, 8]]]).cuda()

        outputs_batch = classifier(inputs_batch)
        print(outputs_batch)