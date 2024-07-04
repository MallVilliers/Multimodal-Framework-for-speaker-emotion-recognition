import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from models.encoder import EncoderLayer
from models.DialogueRNN import DialogueRNN,MatchingAttention,SimpleAttention


class LSTHM(nn.Module):
    def __init__(self, cell_size, in_size, hybrid_in_size):
        super(LSTHM, self).__init__()
        self.cell_size = cell_size
        self.in_size = in_size
        self.W = nn.Linear(in_size, 4*self.cell_size)
        self.U = nn.Linear(cell_size, 4*self.cell_size)
        self.V = nn.Linear(hybrid_in_size, 4*self.cell_size)

    def forward(self, x, ctm, htm, ztm):
        # w*x + u*h + v*z + b
        input_affine = self.W(x)
        output_affine = self.U(htm)
        hybrid_affine = self.V(ztm)

        sums = input_affine + output_affine + hybrid_affine

        # biases are already part of W and U and V
        f_t = torch.sigmoid(sums[:, :self.cell_size])
        i_t = torch.sigmoid(sums[:, self.cell_size:2*self.cell_size])
        o_t = torch.sigmoid(sums[:, 2*self.cell_size:3*self.cell_size])
        ch_t = torch.tanh(sums[:, 3*self.cell_size:])
        c_t = f_t*ctm + i_t*ch_t
        h_t = torch.tanh(c_t)*o_t

        return c_t, h_t


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        [self.d_l, self.d_a, self.d_v] = [100,100,512]
        [self.dh_l, self.dh_a, self.dh_v] = [128,16,64]
        [self.l_reduce_dim, self.a_reduce_dim, self.v_reduce_dim] = [16,128,100]
        self.total_h_dim = self.dh_l + self.dh_a
        self.total_reduce_dim = self.l_reduce_dim + self.a_reduce_dim
        self.num_atts = 4
        output_dim = 6
        # output_dim = 3
        final_out = 2 * self.total_h_dim
        h_out = 32
        out_dropout = 0.0
        map_h = 64
        map_dropout =0.3

        self.nn_out = nn.Sequential(
            nn.Linear(final_out, h_out), 
            nn.ReLU(), 
            nn.Dropout(out_dropout), 
            nn.Linear(h_out, output_dim),
            nn.Softmax(dim=-1))

        d_inner = 50
        n_head = 8
        d_k = 40
        d_v = 40
        self.encoder_l = EncoderLayer(100, d_inner, n_head, d_k, d_v)
        self.encoder_a = EncoderLayer(100, d_inner, n_head, d_k, d_v)

        self.lstm_l = nn.LSTM(self.d_l, self.dh_l, bidirectional=True)
        self.lstm_a = nn.LSTM(self.d_a, self.dh_a, bidirectional=True)

    def forward(self, x):
        # L, B, D
        x_l = x[:, :, :self.d_l].to(x.device).permute(1, 0, 2)
        x_a = x[:, :, self.d_l:self.d_l + self.d_a].to(x.device).permute(1, 0, 2)
        # x_v = x[:, :, self.d_l + self.d_a:].to(x.device).permute(1, 0, 2)
        
        x_l, a_l = self.encoder_l(x_l)    # [30, 79, 100]
        x_a, a_a = self.encoder_a(x_a)    # [30, 79, 100]

        x_l = x_l.permute(1, 0, 2)
        x_a = x_a.permute(1, 0, 2)

        y_l, (h_l, c_l) = self.lstm_l(x_l)  # L, N, D
        y_a, (h_a, c_a) = self.lstm_a(x_a)

        all_hs=torch.cat([y_l, y_a], dim=2).permute(1, 0, 2)
        all_hs = all_hs.reshape(-1, all_hs.size()[-1])
        output=self.nn_out(all_hs)

        
        return output, x_l, x_a




if __name__ == '__main__':
    a=torch.randn((100,200))
    print(a[50:80].shape)