import numpy as np
import torch
from torch import nn

from tts.model.blocks import DurationPredictor


class Predictor(nn.Module):
    def __init__(self, config, regulator_config, encoder_dim, PAD, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(
            config.buckets,
            encoder_dim,
            padding_idx=PAD
        )
        buckets = torch.linspace(np.log1p(config.min_bucket), np.log1p(config.max_bucket+1), config.buckets)
        self.alpha = config.alpha
        self.dur_predictor = DurationPredictor(encoder_dim=encoder_dim,
                                               predictor_filter_size=regulator_config.predictor_filter_size,
                                               predictor_kernel_size=regulator_config.predictor_kernel_size,
                                               dropout=regulator_config.dropout)
        self.register_buffer("buckets", buckets)

    def forward(self, x, target):
        ans = self.dur_predictor(x)
        if target is not None:
            output = self.embedding(torch.bucketize(torch.log1p(target), self.buckets))
        else:
            output = torch.bucketize(torch.log1p((torch.exp(ans)-1)*self.alpha), self.buckets)
            output = self.embedding(torch.clip(output, min=0, max=self.embedding.num_embeddings - 1))
        return output, ans
