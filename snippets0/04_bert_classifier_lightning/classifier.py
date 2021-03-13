from collections import OrderedDict
from typing import List, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import Sequential, CrossEntropyLoss
from torchnlp.utils import lengths_to_mask
from transformers import AutoModel

from snippets0.bert_classifier_lightning.data_module import DataModule
from snippets0.bert_classifier_lightning.utils import mask_fill


class Classifier(pl.LightningModule):
    bert: AutoModel
    tokenizer: Tokenizer
    fc_layers: Sequential

    criterion: CrossEntropyLoss

    frozen = False
    frozen_epochs_count: int

    def __init__(self,
                 batch_size: int,
                 encoder_model: str,
                 frozen_epochs_count: int,
                 loader_workers: int,
                 test_csv: str,
                 train_csv: str,
                 valid_csv: str,
                 ):
        super().__init__()

        self.frozen_epochs_count = frozen_epochs_count

        self.data = DataModule(batch_size, self.prepare_sample, loader_workers,
                               test_csv, train_csv, valid_csv)

        self._build_model(encoder_model, self.data.label_encoder.vocab_size)

        self.criterion = nn.CrossEntropyLoss()

        if self.frozen_epochs_count > 0:
            self.freeze()

    def _build_model(self, encoder_model: str, vocab_size: int):
        self.bert = AutoModel.from_pretrained(encoder_model, output_hidden_states=True)

        self.tokenizer = Tokenizer('bert-base-uncased')

        if encoder_model == 'google/bert_uncased_L-2_H-128_A-2':
            encoder_features = 128
        else:
            encoder_features = 768

        self.fc_layers = nn.Sequential(
            nn.Linear(encoder_features, encoder_features * 2),
            nn.Tanh(),
            nn.Linear(encoder_features * 2, encoder_features),
            nn.Tanh(),
            nn.Linear(encoder_features, vocab_size),
        )

    def forward(self,
                token_batch: torch.IntTensor,
                length_batch: torch.IntTensor
                ) -> torch.FloatTensor:

        token_batch = token_batch[:, :length_batch.max()]

        # When using just one GPU this should not change behavior
        # but when splitting batches across GPUs the tokens have padding
        # from the entire original batch
        mask = lengths_to_mask(length_batch, device=token_batch.device)

        word_embeddings = self.bert(token_batch, mask)[0]

        # Average Pooling
        word_embeddings = mask_fill(0.0, token_batch, word_embeddings, self.tokenizer.padding_index)
        sent_embedding = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        sent_embedding = sent_embedding / sum_mask

        return self.fc_layers(sent_embedding)

    def freeze(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False

        self.frozen = True

    def predict(self, description: str) -> List[str]:
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input = self.pre_process([description])[0]
            model_output = self.forward([model_input]).numpy()[0]

            pred_classes = np.argmax(model_output)
            pred_class_labels = [self.data.label_encoder.index_to_token[pred_class]
                                 for pred_class in pred_classes]

        return pred_class_labels

    def pre_process(self, description_batch: List[str]) -> List:
        pass

    def unfreeze(self) -> None:
        if self.frozen:
            for param in self.bert.parameters():
                param.requires_grad = True

            self.frozen = False

    def training_step(self,
                      input_target_batches: (torch.Tensor, torch.Tensor),
                      batch_count: int
                      ) -> Dict:

        input_batch, target_batch = input_target_batches

        output_batch = self.forward(input_batch)

        loss = self.criterion(output_batch, target_batch)

        # In Data Parallel mode (default) make sure if result is scalar,
        # there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        tqdm_dict = {'train_loss': loss}
        stats = OrderedDict({'loss': loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})

        return stats

    def validation_step(self,
                        input_target_batches: (torch.Tensor, torch.Tensor),
                        batch_count: int
                        ) -> Dict:

        input_batch, target_batch = input_target_batches

        output_batch = self.forward(input_batch)

        loss = self.criterion(output_batch, target_batch)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in Data Parallel mode (default) make sure if result is scalar,
        # there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        stats = OrderedDict({"val_loss": loss, "val_acc": val_acc})

        return stats
