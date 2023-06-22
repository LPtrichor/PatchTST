import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from .layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from .layers.SelfAttention_Family import FullAttention, ProbAttention
# from layers.FED_wo_decomp import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                        out_channels=configs.d_model,
                                        seq_len=self.seq_len,
                                        modes=configs.modes,
                                        mode_select_method=configs.mode_select)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc):
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(x_enc, attn_mask=None)
        return enc_out

class Configs(object):
    ab = 0
    modes = 32
    mode_select = 'random'
    # version = 'Fourier'
    version = 'Wavelets'
    moving_avg = [12, 24]
    L = 1
    base = 'legendre'
    cross_activation = 'tanh'
    seq_len = 96
    label_len = 48
    pred_len = 96
    output_attention = True
    enc_in = 7
    dec_in = 7
    d_model = 512
    embed = 'timeF'
    dropout = 0.05
    freq = 'h'
    factor = 1
    n_heads = 8
    d_ff = 16
    e_layers = 2
    d_layers = 1
    c_out = 7
    activation = 'gelu'
    wavelet = 0

def FEBModule_test(input):
    configs = Configs()
    model = Model(configs)
    x_enc = input
    # x_enc = torch.randn(32, 96, 512)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    out, _ = model.encoder(x_enc)
    return out
    # print(out.shape)

# input = torch.randn(32, 96, 512)
# FEBModule_test(input)


