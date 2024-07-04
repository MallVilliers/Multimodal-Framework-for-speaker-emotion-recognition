import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models.encoder import EncoderLayer, MultiHeadAttention
from attention.ExternalAttention import ExternalAttention


class LSTHM1(nn.Module):
    def __init__(self, cell_size, in_size, hybrid_in_size, speaker_dim):
        super(LSTHM1, self).__init__()
        self.cell_size = cell_size
        self.in_size = in_size
        self.W = nn.Linear(in_size, 4*self.cell_size)
        self.U = nn.Linear(cell_size, 4*self.cell_size)
        self.V = nn.Linear(hybrid_in_size, 4*self.cell_size)
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
        # hybrid_affine = self.V(ztm)     # N, 4*dim
        # speaker_affine=self.S(speaker_affine)

        sums = input_affine + output_affine# N, 4*dim
        
        # biases are already part of W and U and V
        f_t = torch.sigmoid(sums[:, :self.cell_size])   # N, 128
        i_t = torch.sigmoid(sums[:, self.cell_size:2*self.cell_size])   # N, 128
        o_t = torch.sigmoid(sums[:, 2*self.cell_size:3*self.cell_size])     # N, 128
        ch_t = torch.tanh(sums[:, 3*self.cell_size:])       # N, 128
        c_t = f_t * ctm + i_t * ch_t
        h_t = torch.tanh(c_t) * o_t

        return c_t, h_t


class CrossAttention(nn.Module):
    def __init__(self, attn_dropout=0.2):
        super(CrossAttention, self).__init__()
        self.dh = 128

        # 1, D
        self.Wq = nn.Parameter(torch.ones(self.dh).unsqueeze(0))
        self.Wk = nn.Parameter(torch.ones(self.dh).unsqueeze(0))
        self.Wv = nn.Parameter(torch.ones(self.dh).unsqueeze(0))
        
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x_1, x_2):
        # x: B, D
        x_1 = x_1.unsqueeze(-1) # B, D, 1
        x_2 = x_2.unsqueeze(-1) # B, D, 1

        Q = torch.matmul(x_1, self.Wq)   # B, D, D
        K = torch.matmul(x_2, self.Wk)   # B, D, D
        V = x_2 # B, D, 1

        attn = F.softmax(torch.matmul(Q / (self.dh ** 0.5), K) , dim=-1)  # B, D, D
        attn = self.dropout(attn)
        output = torch.matmul(attn, V).squeeze(-1)  # B, D

        return output


class CrossAttention2(nn.Module):
    def __init__(self, dh, dk, dv, attn_dropout=0.2):
        super(CrossAttention2, self).__init__()
        self.dh = dh
        self.dk = dk
        self.dv = dv

        self.Wq = nn.Parameter(torch.ones(self.dh, self.dk))  # D1 * Dk
        self.Wk = nn.Parameter(torch.ones(self.dh, self.dk))  # D2 * Dk
        self.Wv = nn.Parameter(torch.ones(self.dh, self.dv))  # D2 * Dv
        
        self.dropout = nn.Dropout(attn_dropout)

        self.layer_norm = nn.LayerNorm(self.dh, eps=1e-6)

    def forward(self, x_1, x_2):
        # x: L, B, D
        residual = x_1
        x_1 = x_1.permute(1, 0, 2) # B, L1, D1
        x_2 = x_2.permute(1, 0, 2) # B, L2, D2

        Q = torch.matmul(x_1, self.Wq)   # B, L1, Dk
        K = torch.matmul(x_2, self.Wk)   # B, L2, Dk
        V = torch.matmul(x_2, self.Wv)   # B, L2, Dv

        attn = F.softmax(torch.matmul(Q / (self.dk ** 0.5), K.transpose(1, 2)) , dim=-1)  # B, L1, L2
        attn = self.dropout(attn)
        output = torch.matmul(attn, V).permute(1, 0, 2)  # L1, B, Dv

        # add & norm
        output += residual
        output = self.layer_norm(output)

        return output
                

class CrossAttention3(nn.Module):
    def __init__(self, dh, dk, dv, attn_dropout=0.2):
        super(CrossAttention3, self).__init__()
        self.dh = 100
        self.dk = 128
        self.dv = 128

        self.Wq = nn.Parameter(torch.ones(self.dh, self.dk))  # D1 * Dk
        self.Wk = nn.Parameter(torch.ones(self.dk, self.dk))  # D2 * Dk
        self.Wv = nn.Parameter(torch.ones(self.dv, self.dv))  # D2 * Dv
        
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
    def __init__(self, dh_l, dh_a, d_l, d_a, dropout=0.5) -> None:
        super(MARN_cell, self).__init__()
        self.crossatt_l2a = CrossAttention()
        self.crossatt_a2l = CrossAttention()
        self.dh_l, self.dh_a = dh_l, dh_a
        self.dh_q = dh_l
        self.d_l, self.d_a = d_l, d_a
        self.speaker_size = 4 * self.dh_l
        self.dh_s = 128

        self.lsthm_l = LSTHM1(self.dh_l, self.d_l, self.dh_l, self.dh_s)
        self.lsthm_a = LSTHM1(self.dh_a, self.d_a, self.dh_l, self.dh_s)

        self.gru_s = nn.GRUCell(self.d_l + self.d_a, self.dh_s)
        self.gru_l = nn.GRUCell(self.d_l + self.d_a, self.dh_s)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_l, x_a, qmask):

        N = x.shape[1]
        T = x.shape[0]
        h_l = torch.zeros(N, self.dh_l).to(x.device)
        h_a = torch.zeros(N, self.dh_a).to(x.device)
        
        c_l = torch.zeros(N, self.dh_l).to(x.device)
        c_a = torch.zeros(N, self.dh_a).to(x.device)

        z_l = torch.zeros(N, self.dh_l).to(x.device)
        q = torch.zeros(qmask.size()[1], qmask.size()[2], self.dh_s).to(x.device) # batch, party, D_speaker

        h = torch.zeros(0).type(x.type()).to(x.device)

        h_l_lst, h_a_lst, h_sp_lst, h_li_lst = [], [], [], []
        for i in range(T):
            # U = torch.cat((x_l[i], x_a[i]), dim=1)
            U = x[i]
            # 选择当前说话人
            qm_idx = torch.argmax(qmask[i], 1)
            # 上一时刻speaker 和 listener 状态
            qs_0, ql_0 = self._select_parties(q, qm_idx)  # B, D
            # 更新speaker状态
            h_s_ = self.dropout(self.gru_s(U, qs_0))
            # 更新listener状态
            # h_l_ = self.dropout(self.gru_l(U, ql_0))  # B, D
            h_l_ = ql_0
            # 更新q
            q_s = h_s_.unsqueeze(1).expand(-1, qmask[i].size()[1], -1)
            q_l = h_l_.unsqueeze(1).expand(-1, qmask[i].size()[1], -1)  # N, 2, D
            qmask_ = qmask[i].unsqueeze(2) #qmask -> batch, party      N, 2, 1
            q = q_l * (1 - qmask_) + q_s * qmask_      # N, 2, D

            # current time step
            c_l, h_l = self.lsthm_l(x_l[i], *(c_l, h_l, z_l, h_s_))  # B, D
            h_l = self.dropout(h_l)
            c_a, h_a = self.lsthm_a(x_a[i], *(c_a, h_a, z_l, h_s_))  # B, D
            h_a = self.dropout(h_a)

            z_l = self.crossatt_l2a(c_l, c_a)  # B, D
            # # z_a = self.crossatt_a2l(c_a, c_l)   # B, D

            all_hs=torch.cat([h_l, h_a, z_l], dim=1)
            h = torch.cat([h, all_hs.unsqueeze(0)], 0)

            h_l_lst.append(h_l.unsqueeze(0))
            h_a_lst.append(h_a.unsqueeze(0))
            h_sp_lst.append(h_s_.unsqueeze(0))
            h_li_lst.append(h_l_.unsqueeze(0))

        h_l = torch.cat(h_l_lst, dim=0)
        h_a = torch.cat(h_a_lst, dim=0)
        h_sp = torch.cat(h_sp_lst, dim=0)
        h_li = torch.cat(h_li_lst, dim=0)

        return h, h_l, h_a, h_sp, h_li

    def cal_q(self, qtm, qmask, h_s, c_s): 
        qm_idx = torch.argmax(qmask, 1) #本batch各对话当前说话人

        q0_sel = self._select_parties(qtm, qm_idx)  # N, 4*dim
        h_s, c_s = self.lsthm_q(q0_sel, (h_s, c_s))     # N, 4*dim
        q_affine = h_s
        q_s=q_affine.unsqueeze(1).expand(-1,qmask.size()[1],-1)  # N, 2, 4*dim
 
        q_0= q0_sel.unsqueeze(1).expand(-1,qmask.size()[1],-1)  # N, 2, 4*dim
        qmask_ = qmask.unsqueeze(2) #qmask -> batch, party      N, 2, 1
 
        q_t = q_0*(1-qmask_) + q_s*qmask_      # N, 2, 4*dim
        
        return q_t, q_affine, h_s, c_s

    def _select_parties(self, X, indices):
        qs_0, ql_0 = [], []
        for idx, j in zip(indices, X):
            qs_0.append(j[idx].unsqueeze(0))
            ql_0.append(j[1 - idx].unsqueeze(0))
        qs_0 = torch.cat(qs_0, 0)
        ql_0 = torch.cat(ql_0, 0)
        return qs_0, ql_0


class MARN1_nsps(nn.Module):
    def __init__(self, n_classes, dataset):
        super(MARN1_nsps, self).__init__()
        if dataset == 'IEMOCAP':
            self.d_l, self.d_a, self.d_r = 100, 100, 1024
        elif dataset == 'MELD':
            self.d_l, self.d_a, self.d_r = 100, 300, 1024
        self.dh_l, self.dh_a = 128, 128
        self.dh_sp, self.dh_li = 128, 128
        self.total_h_dim = self.dh_l + self.dh_a

        self.linear_in = nn.Linear(self.d_r, self.d_l)  

        self.lstm_l = nn.LSTM(self.d_l, self.dh_l, bidirectional=True)
        self.lstm_a = nn.LSTM(self.d_a, self.dh_a, bidirectional=True)

        output_dim = n_classes
        # final_out = 2 * (self.total_h_dim)
        final_out = self.d_a + self.d_r
        h_out = 32
        out_dropout = 0.5

        self.nn_out = nn.Sequential(
            nn.Linear(final_out, h_out), 
            nn.ReLU(), 
            nn.Dropout(out_dropout), 
            nn.Linear(h_out, output_dim))

        self.dropout_rec = nn.Dropout(0.5)

        self.p = nn.Parameter(torch.ones(2))

        print('++++++++++MARN1_nsps++++++++++')

    def forward(self, x, qmask, umask):
        # x: L, B, D
        # x_l = x[:, :, :self.d_r].to(x.device)
        # x_a = x[:, :, self.d_r:self.d_r + self.d_a].to(x.device)
        # x_l = self.linear_in(x_l)
        x_l, x_a = 0, 0
        # output_l, (h_l, c_l) = self.lstm_l(x_l)     # L, B, D * 2
        # output_a, (h_a, c_a) = self.lstm_a(x_a)     # L, B, D * 2

        output = F.log_softmax(self.nn_out(x), 2)

        output = output.permute(1, 0, 2)
        output = output.reshape(-1, output.size()[-1])
        return output, x_l, x_a

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