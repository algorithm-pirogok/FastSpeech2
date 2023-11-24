import logging
from typing import List

import numpy as np
import torch

from tts.collate_fn.functions import reprocess_tensor


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    batch_expand_size = dataset_items[0]['batch_expand_size']

    len_arr = np.array([d["text"].size(0) for d in dataset_items])
    index_arr = np.argsort(-len_arr)
    batchsize = len(dataset_items)
    real_batchsize = batchsize // batch_expand_size

    cut_list = list()
    for i in range(batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(batch_expand_size):
        output.append(reprocess_tensor(dataset_items, cut_list[i]))
    return output
