import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
from vllm import LLM

"""
Implement all model related functions and classes here.
"""

def get_batch_outputs(model, documents: dict[str, str]) -> dict[str, str]:

    """
        Given a model and a batch of string documents organized as key-value pairs, returns the
        processed and masked documents again as key-value pairs.

        The documents will be given as document name and document contents pairs for keys and values.
    """

    pass
