from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from torch import Tensor
from torchtext.vocab import Vocab

batch_size: int
class_count: int
emb_size: int
sent_count: int
sent_len: int

vocab: Vocab
vocab_size: int


def get_class_lbls():
    return [f'class {i}' for i in range(class_count)]


def get_emb_lbls():
    return [f'emb {i}' for i in range(emb_size)]


def get_ent_class_lbls():
    return [f'ent {i} / class {j}' for i in range(batch_size) for j in range(class_count)]


def get_ent_lbls():
    return [f'ent {i}' for i in range(batch_size)]


def get_ent_sent_lbls():
    return [f'ent {i} / sent {j}' for i in range(batch_size) for j in range(sent_count)]


def get_mix_emb_lbls():
    return [f'mix {i} / class {j}' for i in range(class_count) for j in range(emb_size)]


def get_sent_lbls():
    return [f'sent {i}' for i in range(sent_count)]


def get_tok_lbls():
    return [f'tok {i}' for i in range(sent_len)]


def get_word_lbls():
    return [vocab.itos[i] for i in range(vocab_size)]


def plot_tensor(tensor_, title, labels):
    assert tensor_.ndim <= 4
    assert len(labels) == tensor_.ndim

    min_abs = abs(torch.min(tensor_))
    max_abs = abs(torch.max(tensor_))
    vmin = -max(min_abs, max_abs)
    vmax = max(min_abs, max_abs)

    tensor_ = tensor_.detach()

    if tensor_.ndim == 0:
        tensor_ = np.expand_dims(tensor_, axis=0)
        labels = [['']] + labels

    if tensor_.ndim == 1:
        tensor_ = np.expand_dims(tensor_, axis=0)
        labels = [['']] + labels

    if tensor_.ndim == 2:
        plt.imshow(tensor_, vmin=vmin, vmax=vmax)
        plt.xticks(range(tensor_.shape[1]), labels[1], rotation=45, ha='right')
        plt.yticks(range(tensor_.shape[0]), labels[0])
        plt.colorbar()
        plt.title(title)
        plt.show()

    else:
        if tensor_.ndim == 3:
            tensor_ = np.expand_dims(tensor_, axis=0)
            labels = [['']] + labels

        rows, cols = tensor_.shape[:2]
        fig, axs = plt.subplots(rows, cols, squeeze=False, figsize=(16, 2))
        fig.suptitle(title)

        plt.setp(axs, xticks=range(tensor_.shape[-1]), xticklabels=labels[-1],
                 yticks=range(tensor_.shape[-2]), yticklabels=labels[-2])

        for ax in axs.flat:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        images = []
        for i in range(rows):
            for j in range(cols):
                im = axs[i, j].imshow(tensor_[i, j], vmin=vmin, vmax=vmax)
                images.append(im)
                axs[i, j].label_outer()

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        # Colorbar on right
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(images[0], cax=cbar_ax)

        plt.show()


def log_grad(tensor_: Tensor, title: str, labels: List[List[str]]) -> None:
    assert len(labels) == tensor_.ndim

    grad_title = f'{title}.grad'

    tensor_.register_hook(lambda grad: log_tensor(grad, grad_title, labels))


def log_tensor(tensor_: Tensor, title: str, labels: List[List[str]]) -> None:
    assert len(labels) == tensor_.ndim

    print(title)
    print(tensor_)
    plot_tensor(tensor_, title, labels)
