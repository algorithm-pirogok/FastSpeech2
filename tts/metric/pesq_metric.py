import torch
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

from tts.base.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, fs: int, mode: str, n_processes: int, comb: bool = False, *args, **kwargs):
        self.pesq = PerceptualEvaluationSpeechQuality(fs=fs, mode=mode, n_processes=n_processes)
        self.comb = comb
        super().__init__(*args, **kwargs)

    def __call__(self, **batch):
        short = batch["short"].to("cpu").detach()
        middle = batch["middle"].to("cpu").detach()
        long = batch["long"].to("cpu").detach()
        target = batch["target"].to("cpu").detach()
        pred = short * 0.8 + middle * 0.1 + long * 0.1 if self.comb else short
        try:
            return self.pesq(pred, target)
        except Exception:
            return torch.tensor(-0.5)
