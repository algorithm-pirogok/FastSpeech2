import torch.nn as nn
from tts.base.base_metric import BaseMetric


class CEMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        self.ce = nn.CrossEntropyLoss()
        super().__init__(*args, **kwargs)

    def __call__(self, **batch):
        log_probs = batch["logits"].to("cpu").detach()
        speaker_id = batch["speaker_id"].to("cpu").detach()
        return self.ce(log_probs, speaker_id).item()
