import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models.encoder import EncoderLayer


class LSTHM1(nn.Module):
    def __init__(self, cell_size, in_size, hybrid_in_size, speaker_dim):
        super(LSTHM1, self).__init__()
        self.cell_size = cell_size
        self.in_size = in_size
        self.W = nn.Linear(in_size, 4*self.cell_size)
        self.U = nn.Linear(cell_size, 4*self.cell_size)
        self.V = nn.Linear(hybrid_in_size, self.cell_size)
        self.S = nn.Linear(speaker_dim, 4*self.cell_size)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, x, ctm, htm, ztm, speaker_affine):
        input_affine = self.W(x)        # N, 4*dim
        output_affine = self.U(htm)     # N, 4*dim
        hybrid_affine = self.V(ztm)     # N, 4*dim
        speaker_affine=self.S(speaker_affine)

        sums = input_affine + output_affine + speaker_affine    # N, 4*dim
        
        # biases are already part of W and U and V
        f_t = torch.sigmoid(sums[:, :self.cell_size])   # N, 128
        i_t = torch.sigmoid(sums[:, self.cell_size:2*self.cell_size])   # N, 128
        o_t = torch.sigmoid(sums[:, 2*self.cell_size:3*self.cell_size])     # N, 128
        ch_t = torch.tanh(sums[:, 3*self.cell_size:])       # N, 128
        f_t_ = torch.sigmoid(hybrid_affine) 
        c_t = f_t * ctm + i_t * ch_t + f_t_ * ctm
        h_t = torch.tanh(c_t) * o_t

        return c_t, h_t


class CrossAttention(nn.Module):
    def __init__(self, attn_dropout=0.2):
        super(CrossAttention, self).__init__()
        self.dh = 100
        self.dk = 128
        self.dv = 128

        self.Wq = nn.Parameter(torch.ones(self.dh, self.dk))  # D1 * Dk
        self.Wk = nn.Parameter(torch.ones(self.dh, self.dk))  # D2 * Dk
        self.Wv = nn.Parameter(torch.ones(self.dh, self.dv))  # D2 * Dv
        
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x_1, x_2):
        # x: L, B, D
        x_1 = x_1.permute(1, 0, 2) # B, L1, D1
        x_2 = x_2.permute(1, 0, 2) # B, L2, D2

        Q = torch.matmul(x_1, self.Wq)   # B, L1, Dk
        K = torch.matmul(x_2, self.Wk)   # B, L2, Dk
        V = torch.matmul(x_2, self.Wv)   # B, L2, Dv

        attn = F.softmax(torch.matmul(Q / (self.dk ** 0.5), K.transpose(1, 2)) , dim=-1)  # B, L1, L2
        attn = self.dropout(attn)
        output = torch.matmul(attn, V).permute(1, 0, 2)  # L1, B, Dv

        return output


class MARN_cell(nn.Module):
    def __init__(self, dh_l, dh_a, d_l, d_a) -> None:
        super(MARN_cell, self).__init__()
        self.crossatt_l2a = CrossAttention()
        self.crossatt_a2l = CrossAttention()
        self.dh_l, self.dh_a = dh_l, dh_a
        self.d_l, self.d_a = d_l, d_a
        self.speaker_size = 4 * self.dh_l
        self.dh_s = 128

        self.lsthm_l = LSTHM1(self.dh_l, self.d_l, self.dh_l, self.dh_s)
        self.lsthm_a = LSTHM1(self.dh_a, self.d_a, self.dh_l, self.dh_s)
        self.lstm_s = nn.LSTMCell(self.dh_s, self.dh_s)

        d_inner = 50
        n_head = 8
        d_k = 40
        d_v = 40
        self.encoder_l = EncoderLayer(100, d_inner, n_head, d_k, d_v)
        self.encoder_a = EncoderLayer(100, d_inner, n_head, d_k, d_v)

    def forward(self, x, qmask):
        # x: L, B, D
        x_l = x[:, :, :self.d_l].to(x.device).permute(1, 0, 2)
        x_a = x[:, :, self.d_l:self.d_l + self.d_a].to(x.device).permute(1, 0, 2)
        x_l, a_l = self.encoder_l(x_l)
        x_a, a_a = self.encoder_a(x_a)
        x_l = x_l.permute(1, 0, 2)
        x_a = x_a.permute(1, 0, 2)

        x_la = self.crossatt_l2a(x_l, x_a)  # L, B, D
        x_al = self.crossatt_a2l(x_a, x_l)  # L, B, D

        N = x.shape[1]
        T = x.shape[0]
        h_l = torch.zeros(N, self.dh_l).to(x.device)
        h_a = torch.zeros(N, self.dh_a).to(x.device)
        h_s = torch.zeros(N, self.dh_s).to(x.device)
        c_l = torch.zeros(N, self.dh_l).to(x.device)
        c_a = torch.zeros(N, self.dh_a).to(x.device)
        c_s = torch.zeros(N, self.dh_s).to(x.device)
        z_l = torch.zeros(N, self.dh_l).to(x.device)
        z_a = torch.zeros(N, self.dh_a).to(x.device)
        q = torch.zeros(qmask.size()[1], qmask.size()[2], self.dh_s).to(x.device) # batch, party, D_speaker

        h = torch.zeros(0).type(x.type()).to(x.device)
        for i in range(T):
            q, q_affine, h_s, c_s = self.cal_q(q, qmask[i], h_s, c_s)    # N, 2, 4*dh
            # current time step
            c_l, h_l = self.lsthm_l(x_l[i], *(c_l, h_l, x_la[i], q_affine))  # B, D
            c_a, h_a = self.lsthm_a(x_a[i], *(c_a, h_a, x_al[i], q_affine))  # B, D

            all_hs=torch.cat([h_l, h_a], dim=1)
            h = torch.cat([h, all_hs.unsqueeze(0)], 0)

        return h

    def cal_q(self, qtm, qmask, h_s, c_s):
        qm_idx = torch.argmax(qmask, 1) #本batch各对话当前说话人

        q0_sel = self._select_parties(qtm, qm_idx)  # N, 4*dim
        # q_affine=self.S(q0_sel)     # N, 4*dim
        h_s, c_s = self.lstm_s(q0_sel, (h_s, c_s))     # N, 4*dim
        q_affine = h_s
        q_s=q_affine.unsqueeze(1).expand(-1,qmask.size()[1],-1)  # N, 2, 4*dim
 
        q_0= q0_sel.unsqueeze(1).expand(-1,qmask.size()[1],-1)  # N, 2, 4*dim
        qmask_ = qmask.unsqueeze(2) #qmask -> batch, party      N, 2, 1
 
        q_t = q_0*(1-qmask_) + q_s*qmask_      # N, 2, 4*dim
        
        return q_t, q_affine, h_s, c_s

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        # print(f'q0_sel:{q0_sel}')
        return q0_sel


class MARN1_mf(nn.Module):
    def __init__(self, n_classes):
        super(MARN1_mf, self).__init__()
        self.d_l, self.d_a = 100, 100
        self.dh_l, self.dh_a = 128, 128
        self.total_h_dim = self.dh_l + self.dh_a
        self.speaker_size_l = 4*self.dh_l
        self.speaker_size_a = 4*self.dh_a 

        self.marn_cell_f = MARN_cell(self.dh_l, self.dh_a, self.d_l, self.d_a)
        self.marn_cell_b = MARN_cell(self.dh_l, self.dh_a, self.d_l, self.d_a)

        self.num_atts = 4
        output_dim = n_classes
        final_out = 2 * (self.total_h_dim)
        h_out = 32
        out_dropout = 0.5

        # self.w=nn.Parameter(torch.ones(2))
        self.linear = nn.Linear(final_out, h_out)
        self.nn_out = nn.Sequential(
            nn.Linear(final_out, h_out), 
            nn.ReLU(), 
            nn.Dropout(out_dropout), 
            nn.Linear(h_out, output_dim))

        self.dropout_rec = nn.Dropout(0.5)

    def forward(self, x, qmask, umask):
        # x: L, B, D

        # 正向
        h_f = self.marn_cell_f(x, qmask)    # L, B, D
        h_f = self.dropout_rec(h_f)
        
        # 反向
        rev_x = self._reverse_seq(x, umask)  #倒置输入特征
        
        rev_qmask = self._reverse_seq(qmask, umask)  #倒置本句说话人
        h_b = self.marn_cell_b(rev_x, rev_qmask)  #反方向rnn输出
        h_b = self._reverse_seq(h_b, umask) #把反方向输出倒置
        h_b = self.dropout_rec(h_b)
        h = torch.cat([h_f, h_b], dim=-1)   # L, B, D

        output = F.log_softmax(self.nn_out(h), 2)
        output = output.permute(1, 0, 2)
        output = output.reshape(-1, output.size()[-1])
        return output, 1, 2

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()
        #即每一轮各对话句子数list

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0]) #在第0维翻转（把每一轮各对话进行翻转）
            xfs.append(xf)

        return pad_sequence(xfs)#（后面补零填充）