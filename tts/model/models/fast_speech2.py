import torch
import torch.nn as nn

from tts.base import BaseModel
from tts.model.modules import Encoder, Decoder, LengthRegulator, Predictor
from tts.model.utils import get_mask_from_lengths


class FastSpeech2(BaseModel):
    """ FastSpeech """

    def __init__(self, coder_config, regulator_config, energy_config, pitch_config, max_seq_len, PAD, num_mels, device):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(coder_config, max_seq_len, PAD)
        self.length_regulator = LengthRegulator(regulator_config, coder_config.encoder_dim, device)
        self.decoder = Decoder(coder_config, max_seq_len, PAD)

        self.energy = Predictor(energy_config, regulator_config, coder_config.encoder_dim, PAD)
        self.pitch = Predictor(pitch_config, regulator_config, coder_config.encoder_dim, PAD)

        self.mel_linear = nn.Linear(coder_config.decoder_dim, num_mels)

    @staticmethod
    def mask_tensor(mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, pitch_target=None,
                energy_target=None, alpha=1.0, *args, **kwargs):
        x, non_pad_mask = self.encoder(src_seq, src_pos)

        output, duration_predictor_output = self.length_regulator(x,
                                                                  alpha,
                                                                  length_target=length_target,
                                                                  mel_max_length=mel_max_length)
        pitch_output, pitch_predictor_output = self.pitch(output, pitch_target)
        energy_output, energy_predictor_output = self.energy(pitch_output, energy_target)

        if self.training:
            output = self.decoder(output + pitch_output + energy_output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
        else:
            output = self.decoder(output, mel_pos)

        output = self.mel_linear(output)
        return {"output": output,
                "duration_pred": duration_predictor_output,
                "pitch_pred": pitch_predictor_output,
                "energy_pred": energy_predictor_output}
