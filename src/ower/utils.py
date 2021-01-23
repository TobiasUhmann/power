import torch


def mask_fill(
    fill_value: float,
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """
    Function that masks embeddings representing padded elements.

    :param fill_value: The value to fill the embeddings belonging to padded tokens
    :param tokens: The input sequences [bsz x seq_len]
    :param embeddings: Word embeddings [bsz x seq_len x hiddens]
    :param padding_index: Index of the padding token
    """

    padding_mask = tokens.eq(padding_index).unsqueeze(-1)

    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)
