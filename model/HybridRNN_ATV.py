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


class MARN(nn.Module):
    def __init__(self):
        super(MARN, self).__init__()
        [self.d_l, self.d_a, self.d_v] = [100,100,512]
        [self.dh_l, self.dh_a, self.dh_v] = [128,16,64]
        [self.l_reduce_dim, self.a_reduce_dim, self.v_reduce_dim] = [16,128,100]
        self.total_h_dim = self.dh_l + self.dh_a + self.dh_v
        self.total_reduce_dim = self.l_reduce_dim + self.a_reduce_dim + self.v_reduce_dim
        self.num_atts = 4
        output_dim = 6
        # output_dim = 3
        final_out = 2 * self.total_h_dim
        h_out = 32
        out_dropout = 0.0
        map_h = 64
        map_dropout =0.3
        self.lsthm_l = LSTHM(self.dh_l, self.d_l, self.total_h_dim)
        self.lsthm_a = LSTHM(self.dh_a, self.d_a, self.total_h_dim)
        self.lsthm_v = LSTHM(self.dh_v, self.d_v, self.total_h_dim)

        self.att = nn.Sequential(nn.Linear(self.total_h_dim, self.num_atts * self.total_h_dim))

        self.reduce_dim_nn_l = nn.Sequential(nn.Linear(self.num_atts * self.dh_l, self.l_reduce_dim))
        self.reduce_dim_nn_a = nn.Sequential(nn.Linear(self.num_atts * self.dh_a, self.a_reduce_dim))
        self.reduce_dim_nn_v = nn.Sequential(nn.Linear(self.num_atts * self.dh_v, self.v_reduce_dim))

        self.fc = nn.Sequential(nn.Linear(self.total_reduce_dim, map_h), nn.ReLU(), nn.Dropout(map_dropout), nn.Linear(map_h, self.total_h_dim))

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
        self.encoder_v = EncoderLayer(512, d_inner, n_head, d_k, d_v)


    def forward(self, x):
        # L, B, D
        x_l = x[:, :, :self.d_l].to(x.device).permute(1, 0, 2)
        x_a = x[:, :, self.d_l:self.d_l + self.d_a].to(x.device).permute(1, 0, 2)
        x_v = x[:, :, self.d_l + self.d_a:].to(x.device).permute(1, 0, 2)
        
        x_l, a_l = self.encoder_l(x_l)    # [30, 79, 100]
        x_a, a_a = self.encoder_a(x_a)    # [30, 79, 100]
        x_v, a_v = self.encoder_v(x_v)    # [30, 79, 512]

        x_l = x_l.permute(1, 0, 2)
        x_a = x_a.permute(1, 0, 2)
        x_v = x_v.permute(1, 0, 2)


        # x is T*N*d
        N = x.shape[1]
        T = x.shape[0]
        self.h_l = torch.zeros(N, self.dh_l).to(x.device)
        self.h_a = torch.zeros(N, self.dh_a).to(x.device)
        self.h_v = torch.zeros(N, self.dh_v).to(x.device)
        self.c_l = torch.zeros(N, self.dh_l).to(x.device)
        self.c_a = torch.zeros(N, self.dh_a).to(x.device)
        self.c_v = torch.zeros(N, self.dh_v).to(x.device)
        self.z_t = torch.zeros(N, self.total_h_dim).to(x.device)
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_z_ts = []
        output=[]
        for i in range(T):
            # current time step
            new_c_l, new_h_l  = self.lsthm_l(x_l[i], *(self.c_l, self.h_l, self.z_t))
            new_c_a, new_h_a  = self.lsthm_a(x_a[i], *(self.c_a, self.h_a, self.z_t))
            new_c_v, new_h_v  = self.lsthm_v(x_v[i], *(self.c_v, self.h_v, self.z_t))

            new_cs = torch.cat([new_c_l, new_c_a, new_c_v], dim=1)
            attention = F.softmax(torch.cat(torch.chunk(self.att(new_cs), self.num_atts, dim=1), dim=0), dim=1)
            attended = attention * new_cs.repeat(self.num_atts, 1)
            reduce_l = self.reduce_dim_nn_l(torch.cat(torch.chunk(attended[:, :self.dh_l], self.num_atts, dim=0), dim=1))
            reduce_a = self.reduce_dim_nn_a(torch.cat(torch.chunk(attended[:, self.dh_l:self.dh_l+self.dh_a], self.num_atts, dim=0), dim=1))
            reduce_v = self.reduce_dim_nn_v(torch.cat(torch.chunk(attended[:, self.dh_l+self.dh_a:], self.num_atts, dim=0), dim=1))
            self.z_t = self.fc(torch.cat([reduce_l, reduce_a, reduce_v], dim=1))
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)
            all_z_ts.append(self.z_t)
            all_hs=torch.cat([self.h_l, self.h_a, self.h_v, self.z_t], dim=1)
            out=self.nn_out(all_hs)
            # print(out.shape)
            output.append(out)

        # last_h_l = all_h_ls[-1]
        # last_h_a = all_h_as[-1]
        # last_h_v = all_h_vs[-1]
        # last_z_t = all_z_ts[-1]
        # last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_z_t], dim=1)
        # output = self.nn_out(last_hs)
        # print(f'output:{len(output)}')

        output=torch.cat(output,dim=0)
        
        return output


# class HybridRNNCell(nn.Module):

#     def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
#                             context_attention='simple', D_a=100, dropout=0.5):
#         super(HybridRNNCell, self).__init__()

#         self.D_m = D_m
#         self.D_g = D_g
#         self.D_p = D_p
#         self.D_e = D_e

#         self.listener_state = listener_state
#         self.g_cell = nn.GRUCell(D_m+D_p,D_g)
#         self.p_cell = nn.GRUCell(D_m+D_g,D_p)
#         self.e_cell = nn.GRUCell(D_p,D_e)
#         if listener_state:
#             self.l_cell = nn.GRUCell(D_m+D_p,D_p)

#         self.dropout = nn.Dropout(dropout)

#         if context_attention=='simple':
#             self.attention = SimpleAttention(D_g)
#         else:
#             self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

#     def _select_parties(self, X, indices):
#         q0_sel = []
#         for idx, j in zip(indices, X):
#             q0_sel.append(j[idx].unsqueeze(0))
#         q0_sel = torch.cat(q0_sel,0)
#         # print(f'q0_sel:{q0_sel}')
#         return q0_sel

#     def forward(self, U, qmask, g_hist, q0, e0):
#         """
#         U -> batch, D_m
#         qmask -> batch, party
#         g_hist -> t-1, batch, D_g
#         q0 -> batch, party, D_p
#         e0 -> batch, self.D_e
#         """
        
#         qm_idx = torch.argmax(qmask, 1) #本batch各对话当前说话人
#         # print(f'qm_idx{qm_idx}')
#         q0_sel = self._select_parties(q0, qm_idx)

#         g_ = self.g_cell(torch.cat([U,q0_sel], dim=1),
#                 torch.zeros(U.size()[0],self.D_g).type(U.type()).to(U.device) if g_hist.size()[0]==0 else
#                 g_hist[-1])
#         g_ = self.dropout(g_)
#         if g_hist.size()[0]==0:
#             c_ = torch.zeros(U.size()[0],self.D_g).type(U.type()).to(U.device)
#             alpha = None
#         else:
#             c_, alpha = self.attention(g_hist,U)
#         # c_ = torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0\
#         #         else self.attention(g_hist,U)[0] # batch, D_g
#         U_c_ = torch.cat([U,c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
#         qs_ = self.p_cell(U_c_.contiguous().view(-1,self.D_m+self.D_g),
#                 q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
#         qs_ = self.dropout(qs_)

#         if self.listener_state:
#             U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
#             ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
#                     expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_p)
#             U_ss_ = torch.cat([U_,ss_],1)
#             ql_ = self.l_cell(U_ss_,q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
#             ql_ = self.dropout(ql_)
#         else:
#             ql_ = q0
#         qmask_ = qmask.unsqueeze(2)
#         q_ = ql_*(1-qmask_) + qs_*qmask_
#         e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()).to(U.device) if e0.size()[0]==0\
#                 else e0
#         e_ = self.e_cell(self._select_parties(q_,qm_idx), e0)
#         e_ = self.dropout(e_)

#         return g_,q_,e_,alpha

# class HybridRNN(nn.Module):

#     def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
#                             context_attention='simple', D_a=100, dropout=0.5):
#         super(HybridRNN, self).__init__()

#         self.D_m = D_m
#         self.D_g = D_g
#         self.D_p = D_p
#         self.D_e = D_e
#         self.dropout = nn.Dropout(dropout)

#         self.hcell1 = HybridRNNCell(100, D_g, D_p, D_e,
#                             listener_state, context_attention, D_a, dropout)
#         self.hcell2 = HybridRNNCell(100, D_g, D_p, D_e,
#                             listener_state, context_attention, D_a, dropout)
#         self.hcell3 = HybridRNNCell(512, D_g, D_p, D_e,
#                             listener_state, context_attention, D_a, dropout)

#     def forward(self, U, qmask):
#         """bsdbcdbcd
#         U -> seq_len, batch, D_m
#         qmask -> seq_len, batch, party
#         """

#         g_hist = torch.zeros(0).type(U.type()).to(U.device) # 0-dimensional tensor
#         q_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_p).type(U.type()).to(U.device) # batch, party, D_p
#         e_ = torch.zeros(0).type(U.type()).to(U.device) # batch, D_e
#         e = e_

#         alpha = []
#         # i=0
#         for u_,qmask_ in zip(U, qmask): 
#             # print(i)
#             # i+=1
#             # print(f'u_:{u_.shape}')
#             # print(f'u[0:100]:{u_[:,:100].shape}')
#             g1_, q1_, e1_, alpha1_ = self.hcell1(u_[:,:100], qmask_, g_hist, q_, e_,hy_)
#             g2_, q2_, e2_, alpha2_ = self.hcell2(u_[:,100:200], qmask_, g_hist, q_, e_)
#             g3_, q3_, e3_, alpha3_ = self.hcell3(u_[:,200:], qmask_, g_hist, q_, e_)
#             q_=q1_+q2_+q3_
#             e_=e1_+e2_+e3_
#             g_=g1_+g2_+g3_
#             if type(alpha1_)!=type(None):
#                 alpha_=alpha1_+alpha2_+alpha3_
#             else:
#                 alpha_=None
#             g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
#             e = torch.cat([e, e_.unsqueeze(0)],0)
#             if type(alpha_)!=type(None):
#                 alpha.append(alpha_[:,0,:])

#         return e,alpha # seq_len, batch, D_e


# class HyModel(nn.Module):

#     def __init__(self, D_m, D_g, D_p, D_e, D_h,
#                  n_classes=7, listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5,
#                  dropout=0.5):
#         super(HyModel, self).__init__()

#         self.D_m       = D_m
#         self.D_g       = D_g
#         self.D_p       = D_p
#         self.D_e       = D_e
#         self.D_h       = D_h
#         self.n_classes = n_classes
#         self.dropout   = nn.Dropout(dropout)
#         self.dropout_rec = nn.Dropout(dropout+0.15)
#         # self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e,listener_state,
#         #                             context_attention, D_a, dropout_rec)
#         # self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e,listener_state,
#         #                             context_attention, D_a, dropout_rec)
#         self.dialog_rnn_f = HybridRNN(D_m, D_g, D_p, D_e,listener_state,
#                                     context_attention, D_a, dropout_rec)
#         self.dialog_rnn_r = HybridRNN(D_m, D_g, D_p, D_e,listener_state,
#                                     context_attention, D_a, dropout_rec)
#         self.linear     = nn.Linear(2*D_e, 2*D_h)
#         self.smax_fc    = nn.Linear(2*D_h, n_classes)
#         self.matchatt = MatchingAttention(2*D_e,2*D_e,att_type='general2')

#     def _reverse_seq(self, X, mask):
#         """
#         X -> seq_len, batch, dim
#         mask -> batch, seq_len
#         """
#         X_ = X.transpose(0,1)
#         mask_sum = torch.sum(mask, 1).int()
#         #即每一轮各对话句子数list
#         # print(f'mask_sum:{mask_sum}')

#         xfs = []
#         for x, c in zip(X_, mask_sum):
#             xf = torch.flip(x[:c], [0]) #在第0维翻转（把每一轮各对话进行翻转）
#             xfs.append(xf)

#         return pad_sequence(xfs)#（后面补零填充）


#     def forward(self, U, qmask, umask,att2=True):
#         """
#         U -> seq_len, batch, D_m example:torch.size[69,30,100],batchsize=30,对话中最长有69个句子，每个句子用100维度表示
#         qmask -> seq_len, batch, party #对话最长69个句子，batchsize=30,本句说话人的one hot编码
#         umask:是对话句子数的mask(屏蔽向量的填充部分)
#         """

#         emotions_f, alpha_f = self.dialog_rnn_f(U, qmask) # seq_len, batch, D_e
#         emotions_f = self.dropout_rec(emotions_f)
#         rev_U = self._reverse_seq(U, umask)  #倒置输入特征
#         # print(f'umask:{umask.shape}')
        
#         rev_qmask = self._reverse_seq(qmask, umask)  #倒置本句说话人
#         emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)  #反方向rnn输出
#         emotions_b = self._reverse_seq(emotions_b, umask) #把反方向输出倒置
#         emotions_b = self.dropout_rec(emotions_b)
#         emotions = torch.cat([emotions_f,emotions_b],dim=-1)
#         # print(f'emotions:{emotions.shape}')
#         if att2:
#             att_emotions = []
#             alpha = []
#             for t in emotions:  #t：batch中各对话第i个句子的表示张量
#                 att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
#                 att_emotions.append(att_em.unsqueeze(0)) # batch, mem_dim->1,batch,mem_dim
#                 alpha.append(alpha_[:,0,:])
#             att_emotions = torch.cat(att_emotions,dim=0)
#             hidden = F.relu(self.linear(att_emotions))
#         else:
#             hidden = F.relu(self.linear(emotions))
#         #hidden = F.relu(self.linear(emotions))
#         hidden = self.dropout(hidden)
#         # log_prob = hidden midmodel latemodel用这个
#         log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes
#         if att2:
#             return log_prob, alpha, alpha_f, alpha_b
#         else:
#             return log_prob, [], alpha_f, alpha_b

# if __name__ == '__main__':
#     a=torch.randn((100,200))
#     print(a[50:80].shape)