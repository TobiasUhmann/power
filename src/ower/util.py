import logging

from torch import Tensor


def log_tensor(module_path: str, tensor_str: str, tensor: Tensor, shape_str: str) -> None:
    logging.debug(
        f'{module_path}' '\n'
        f'{tensor_str} {shape_str}' '\n'
        f'{tensor}')

    if tensor.requires_grad:
        tensor.register_hook(lambda grad: logging.debug(
            f'{module_path}' '\n'
            f'{tensor_str}.grad {shape_str}' '\n'
            f'{grad}'))
