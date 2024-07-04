import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models.encoder import EncoderLayer
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
        hybrid_affine = self.V(ztm)     # N, 4*dim
        speaker_affine=self.S(speaker_affine)

        sums = input_affine + output_affine + hybrid_affine + speaker_affine# N, 4*dim
        
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
        # self.lsthm_q0 = LSTHM1(self.dh_a, self.dh_s, self.dh_l, self.dh_s)
        # self.lsthm_q1 = LSTHM1(self.dh_a, self.dh_s, self.dh_l, self.dh_s)
        self.lstm_q0 = nn.LSTMCell(self.dh_s, self.dh_s)
        self.lstm_q1 = nn.LSTMCell(self.dh_s, self.dh_s)

        # self.lsthm_q0 = nn.Linear(self.dh_s, self.dh_s)
        # self.lsthm_q1 = nn.Linear(self.dh_s, self.dh_s)

        self.lstm_s = nn.LSTMCell(self.dh_s, self.dh_s)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_l, x_a, qmask):

        N = x.shape[1]
        T = x.shape[0]
        h_l = torch.zeros(N, self.dh_l).to(x.device)
        h_a = torch.zeros(N, self.dh_a).to(x.device)
        h_q0 = torch.zeros(N, self.dh_q).to(x.device)
        h_q1 = torch.zeros(N, self.dh_q).to(x.device)
        
        c_l = torch.zeros(N, self.dh_l).to(x.device)
        c_a = torch.zeros(N, self.dh_a).to(x.device)
        c_q0 = torch.zeros(N, self.dh_q).to(x.device)
        c_q1 = torch.zeros(N, self.dh_q).to(x.device)

        z_l = torch.zeros(N, self.dh_l).to(x.device)
        z_a = torch.zeros(N, self.dh_a).to(x.device)
        q = torch.zeros(qmask.size()[1], qmask.size()[2], self.dh_s).to(x.device) # batch, party, D_speaker

        h = torch.zeros(0).type(x.type()).to(x.device)
        for i in range(T):
            # 选择当前说话人状态
            qm_idx = torch.argmax(qmask[i], 1) #本batch各对话当前说话人
            q0_sel, q1_sel, N0, N1 = self._select_parties(q, qm_idx)  # B, D
            # 更新当前说话人状态
            if N0:
                # c_q0, h_q0 = self.lsthm_q0(q0_sel, *(c_q0, h_q0, z_l, q0_sel))  # B, D
                h_q0, c_q0 = self.lstm_q0(q0_sel, (h_q0, c_q0))  
                h_q0 = self.dropout(h_q0)
                # h_q0 = self.lsthm_q0(q0_sel)
            if N1:
                # c_q1, h_q1 = self.lsthm_q1(q1_sel, *(c_q1, h_q1, z_l, q1_sel))  # B, D
                h_q1, c_q1 = self.lstm_q1(q1_sel, (h_q1, c_q1))  # B, D
                h_q1 = self.dropout(h_q1)
                # h_q1 = self.lsthm_q0(q1_sel)
            # 更新q
            if N0 and N1:
                h_q = torch.cat([h_q0[: N0], h_q1[: N1]], dim=0)
                h_0 = torch.cat([q0_sel[: N0], q1_sel[: N1]], dim=0)
                # h_q = torch.cat([h_q0, h_q1], dim=0)
                # h_0 = torch.cat([q0_sel, q1_sel], dim=0)
            elif N0:
                h_q = h_q0
                h_0 = q0_sel
            else:
                h_q = h_q1
                h_0 = q1_sel


            q_s=h_q.unsqueeze(1).expand(-1, qmask[i].size()[1], -1)
            q_0= h_0.unsqueeze(1).expand(-1, qmask[i].size()[1], -1)  # N, 2, D
            qmask_ = qmask[i].unsqueeze(2) #qmask -> batch, party      N, 2, 1
            q = q_0 * (1 - qmask_) + q_s * qmask_      # N, 2, D

            # current time step
            c_l, h_l = self.lsthm_l(x_l[i], *(c_l, h_l, z_l, h_q))  # B, D
            h_l = self.dropout(h_l)
            c_a, h_a = self.lsthm_a(x_a[i], *(c_a, h_a, z_l, h_q))  # B, D
            h_a = self.dropout(h_a)

            z_l = self.crossatt_l2a(c_l, c_a)  # B, D
            # # z_a = self.crossatt_a2l(c_a, c_l)   # B, D

            all_hs=torch.cat([h_l, h_a, z_l, h_q], dim=1)
            h = torch.cat([h, all_hs.unsqueeze(0)], 0)

        return h

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
        N = X.size()[0]
        q0_sel, q1_sel = [], []

        for idx, j in zip(indices, X):
            if idx == 0:     
                q0_sel.append(j[idx].unsqueeze(0))
            elif idx == 1:
                q1_sel.append(j[idx].unsqueeze(0))
        N0, N1 = len(q0_sel), len(q1_sel)

        if N0:        
            q0_sel = torch.cat(q0_sel, 0)
            q0_zero = torch.zeros(N - N0, q0_sel.size()[1]).to(q0_sel.device)
            q0_sel = torch.cat((q0_sel, q0_zero), 0)

        if N1:
            q1_sel = torch.cat(q1_sel, 0)
            q1_zero = torch.zeros(N - N1, q1_sel.size()[1]).to(q1_sel.device)
            q1_sel = torch.cat((q1_sel, q1_zero), 0)
        
        return q0_sel, q1_sel, N0, N1

    def circulant_matrix(self, l, a):
        B, L = l.size()
        l = l.unsqueeze(2)
        a = a.unsqueeze(2)
        l_vecs, a_vecs = [], []
        for i in range(L):
            pre_l = l[:, L-i:, :]
            tail_l = l[:, 0: L-i, :]
            vec_l = torch.cat([pre_l, tail_l], dim=1)
            l_vecs.append(vec_l)

            pre_a = a[:, L-i:, :]
            tail_a = a[:, 0: L-i, :]
            vec_a = torch.cat([pre_a, tail_a], dim=1)
            a_vecs.append(vec_a)

        mat_l = torch.cat(l_vecs, dim=2)
        mat_a = torch.cat(a_vecs, dim=2)

        return mat_l, mat_a

    def fusion_matrix_multiplication(self, l, a):

        l_matrix, a_matrix = self.circulant_matrix(l, a)    # B, D, D

        l = l.unsqueeze(2)  # B, D, 1
        a = a.unsqueeze(2)  # B, D, 1

        fusion_l = torch.matmul(l_matrix, a).squeeze(2)    # B, D
        fusion_a = torch.matmul(a_matrix, l).squeeze(2)    # B, D

        fusion_l = F.relu(fusion_l)
        fusion_a = F.relu(fusion_a)

        return fusion_l, fusion_a


class MARN1_sps(nn.Module):
    def __init__(self, n_classes):
        super(MARN1_sps, self).__init__()
        self.d_l, self.d_a, self.d_r = 100, 100, 1024
        self.dh_l, self.dh_a = 128, 128
        self.dh_sp, self.dh_li = 128, 128
        self.total_h_dim = self.dh_l + self.dh_a

        self.linear_in = nn.Linear(self.d_r, self.d_l)
        self.marn_cell_f = MARN_cell(self.dh_l, self.dh_a, self.d_l, self.d_a)
        self.marn_cell_b = MARN_cell(self.dh_l, self.dh_a, self.d_l, self.d_a)

        output_dim = n_classes
        final_out = 2 * (self.total_h_dim + self.dh_l+ self.dh_l) + self.dh_l + self.dh_a
        # final_out = 2 * (self.total_h_dim) + 128
        h_out = 32
        out_dropout = 0.5
        self.fc = nn.Sequential(
            nn.Linear(final_out, self.d_l), 
            nn.ReLU(), 
            nn.Dropout(out_dropout))

        self.nn_out = nn.Sequential(
            nn.Linear(self.d_l, h_out), 
            nn.ReLU(), 
            nn.Dropout(out_dropout), 
            nn.Linear(h_out, output_dim))

        self.dropout_rec = nn.Dropout(0.5)
        
        # encoder
        d_inner = 40
        n_head = 8
        d_k = 40
        d_v = 40
        self.encoder_l = EncoderLayer(100, d_inner, n_head, d_k, d_v)
        self.encoder_a = EncoderLayer(100, d_inner, n_head, d_k, d_v)
        self.crossatt_l2a = CrossAttention2(self.d_l, self.dh_l, self.dh_l)
        self.crossatt_a2l = CrossAttention2(self.d_a, self.dh_a, self.dh_a)
        self.crossatt_l2a_1 = CrossAttention3(self.dh_l, self.d_l, self.d_l)
        self.crossatt_a2l_1 = CrossAttention3(self.dh_a, self.d_a, self.d_a)

        self.w = nn.Parameter(torch.ones(1))
        self.v = nn.Parameter(torch.ones(1))
        # self.p = nn.Parameter(torch.ones(3))
        # self.w1 = nn.Parameter(torch.ones(1))
        self.v1 = nn.Parameter(torch.ones(1))
        # self.w2 = nn.Parameter(torch.ones(1))
        self.v2 = nn.Parameter(torch.ones(1))


    def forward(self, x, qmask, umask):
        # x: L, B, D
        x_l = x[:, :, :self.d_r].to(x.device).permute(1, 0, 2)
        x_a = x[:, :, self.d_r:self.d_r + self.d_a].to(x.device).permute(1, 0, 2)
        x_l = self.linear_in(x_l)
        x_l_1, a_l = self.encoder_l(x_l)
        x_a_1, a_a = self.encoder_a(x_a)

        x_l, a_l = self.encoder_l(x_l + x_l_1)
        x_a, a_a = self.encoder_a(x_a + x_a_1)

        x_l = x_l.permute(1, 0, 2)  
        x_a = x_a.permute(1, 0, 2)

        # 正向
        h_f = self.marn_cell_f(x, x_l, x_a, qmask)    # L, B, D
        h_f = self.dropout_rec(h_f)
        
        # 反向
        rev_x_l = self._reverse_seq(x_l, umask) 
        rev_x_a = self._reverse_seq(x_a, umask) 
        
        rev_qmask = self._reverse_seq(qmask, umask)  #倒置本句说话人
        h_b = self.marn_cell_b(x, rev_x_l, rev_x_a, rev_qmask)  #反方向rnn输出
        h_b = self._reverse_seq(h_b, umask) #把反方向输出倒置
        h_b = self.dropout_rec(h_b)
        h = torch.cat([h_f, h_b], dim=-1)   # L, B, D

        attn1 = self.crossatt_l2a(self.w * x_l, self.v * x_a)   # L, B, D
        attn2 = self.crossatt_a2l(self.v * x_a, self.w * x_l)   # L, B, D
        # attn1 = self.crossatt_l2a(x_l, x_a)   # L, B, D
        # attn2 = self.crossatt_a2l(x_a, x_l)   # L, B, D

        attn1 = self.crossatt_l2a_1(self.v * x_a, self.v1 * attn1)   # L, B, D
        attn2 = self.crossatt_a2l_1(self.w * x_l, self.v2 * attn2)   # L, B, D

        # w1 = torch.exp(self.p[0]) / torch.sum(torch.exp(self.p))
        # w2 = torch.exp(self.p[1]) / torch.sum(torch.exp(self.p))
        # w3 = torch.exp(self.p[2]) / torch.sum(torch.exp(self.p))
        # w4 = torch.exp(self.p[3]) / torch.sum(torch.exp(self.p))
        # w5 = torch.exp(self.p[4]) / torch.sum(torch.exp(self.p))
        output = self.fc(torch.cat([h, attn1, attn2], dim=-1))
        output = F.log_softmax(self.nn_out(output + x_l + x_a), 2)
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