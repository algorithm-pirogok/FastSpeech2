from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from tts.base.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, zero_mean: bool, comb: bool, *args, **kwargs):
        self.si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=zero_mean)
        self.comb = comb
        super().__init__(*args, **kwargs)

    def __call__(self, **batch):
        short = batch["short"].to("cpu").detach()
        middle = batch["middle"].to("cpu").detach()
        long = batch["long"].to("cpu").detach()
        target = batch["target"].to("cpu").detach()
        pred = short * 0.8 + middle * 0.1 + long * 0.1 if self.comb else short
        return self.si_sdr(pred, target)
