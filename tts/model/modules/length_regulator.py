import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.model.blocks import DurationPredictor
from tts.model.utils import create_alignment


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, regulator_config, encoder_dim, device):
        super(LengthRegulator, self).__init__()

        self.duration_predictor = DurationPredictor(encoder_dim, regulator_config.predictor_filter_size,
                                                    regulator_config.predictor_kernel_size,
                                                    regulator_config.dropout)
        self.device = device

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, length_target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if length_target is not None:
            output = self.LR(x, length_target, mel_max_length)
            return output, duration_predictor_output

        duration_predictor_output = F.relu((torch.exp(duration_predictor_output)-1) * alpha).int()
        output = self.LR(x, duration_predictor_output)
        mel_pos = torch.vstack([torch.Tensor([i + 1 for i in range(output.size(1))])]).long().to(self.device)

        return output, mel_pos
