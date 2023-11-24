from torch import nn

from tts.model.blocks import FFTBlock
from tts.model.utils import get_non_pad_mask, get_attn_key_pad_mask


class Encoder(nn.Module):
    def __init__(self, encoder_config, max_seq_len, PAD):
        super(Encoder, self).__init__()

        len_max_seq = max_seq_len
        n_position = len_max_seq + 1
        n_layers = encoder_config.encoder_n_layer

        self.PAD = PAD

        self.src_word_emb = nn.Embedding(
            encoder_config.vocab_size,
            encoder_config.encoder_dim,
            padding_idx=PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            encoder_config.encoder_dim,
            padding_idx=PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            encoder_config.encoder_dim,
            encoder_config.encoder_conv1d_filter_size,
            encoder_config.encoder_head,
            encoder_config.encoder_dim // encoder_config.encoder_head,
            encoder_config.encoder_dim // encoder_config.encoder_head,
            encoder_config.fft_conv1d_kernel,
            encoder_config.fft_conv1d_padding,
            dropout=encoder_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, PAD=self.PAD)
        non_pad_mask = get_non_pad_mask(src_seq, self.PAD)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask
