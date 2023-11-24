from typing import List

import torch
from torch import Tensor

from tts.base.base_metric import BaseMetric
from tts.base.base_text_encoder import BaseTextEncoder
from tts.metric.utils import calc_cer


class CERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, mode: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.mode = mode

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []

        lengths = log_probs_length.detach().numpy()
        if self.mode == 'argmax':
            predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
            for log_prob_vec, length, target_text in zip(predictions, lengths, text):
                target_text = BaseTextEncoder.normalize_text(target_text)
                if hasattr(self.text_encoder, "ctc_decode"):
                    pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
                else:
                    pred_text = self.text_encoder.decode(log_prob_vec[:length])
                cers.append(calc_cer(target_text, pred_text))
        elif self.mode == 'beam-search':
            if not hasattr(self.text_encoder, "ctc_beam_search"):
                raise Exception('Where is beam search?')
            beam_search = getattr(self.text_encoder, "ctc_beam_search")
            for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
                pred_text = beam_search(log_prob_vec, length, **kwargs)[0].text
                target_text = BaseTextEncoder.normalize_text(target_text)
                cers.append(calc_cer(target_text, pred_text))
        elif self.mode == 'lm':
            if not hasattr(self.text_encoder, "ctc_beam_search"):
                raise Exception('Where is lm?')
            lm = getattr(self.text_encoder, "lm_beam_search")
            for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
                pred_text = lm(log_prob_vec, length, **kwargs)
                target_text = BaseTextEncoder.normalize_text(target_text)
                cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
