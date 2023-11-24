import os.path
from collections import defaultdict
from typing import List, NamedTuple

import shutil
import wget
import torch
from pyctcdecode import build_ctcdecoder
import gzip

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, path_to_lm: str = "data/lm/language_model.arpa"):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.decoding_mode = "ctc" if path_to_lm is None else "lm"
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        results = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            if not ind:
                continue
            if last_char != self.ind2char[ind]:
                results.append(self.ind2char[ind])
            last_char = self.ind2char[ind]
        return ''.join(results)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 1, **kwargs) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        probs = probs[:probs_length]
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        def _extend_and_merge(frame_distr, state_dict):
            new_state_dict = defaultdict(float)
            for next_char_index, next_char_proba in enumerate(frame_distr):
                for (pref, last_char), pref_proba in state_dict.items():
                    next_char = self.ind2char[next_char_index]
                    if next_char == last_char:
                        new_pref = pref
                    else:
                        if next_char != self.EMPTY_TOK:
                            new_pref = pref + next_char
                        else:
                            new_pref = pref
                        last_char = next_char
                    new_state_dict[(new_pref, last_char)] += pref_proba * next_char_proba
            return new_state_dict

        def _truncate(state_dict):
            state_list = list(state_dict.items())
            return dict(sorted(state_list, key=lambda x: -x[1].item())[:beam_size])

        state = {('', self.EMPTY_TOK): 1.0}
        for frame in probs:
            state = _extend_and_merge(frame, state)
            state = _truncate(state)

        hypos: List[Hypothesis] = [Hypothesis(seq, prob) for (seq, _), prob in state.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def lm_batch_beam_search(self, probs, probs_length, multy_pool, size_of_beam_search=50, **kwargs):
        batch = [prob[:length].detach().cpu().numpy() for prob, length in zip(probs, probs_length)]
        return self.decoder.language_model.decode_batch(multy_pool, batch, size_of_beam_search)

    def lm_beam_search(self, probs: torch.tensor, probs_length, **kwargs):
        assert len(probs.shape) == 2
        probs = probs[:probs_length].detach().cpu().numpy()
        ans = self.decoder.language_model.decode(probs, beam_width=40).lower()
        return ans
