import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import EncoderLayer
from models.lsthm_h import LSTHM1, general_attention


class MARN1_newz(nn.Module):
    def __init__(self):
        super(MARN1_newz, self).__init__()
        [self.d_l, self.d_a] = [100, 100]
        [self.dh_l, self.dh_a] = [128, 128]
        [self.l_reduce_dim, self.a_reduce_dim] = [16, 128]
        self.total_h_dim = self.dh_l + self.dh_a
        self.total_reduce_dim = self.l_reduce_dim + self.a_reduce_dim
        self.speaker_size_l=4*self.dh_l
        self.speaker_size_a=4*self.dh_a

        self.num_atts = 4
        output_dim = 6
        # output_dim = 3
        final_out = 2 * self.total_h_dim
        h_out = 32
        out_dropout = 0.5
        map_h = 64
        map_dropout =0.3
        self.lsthm_l = LSTHM1(self.dh_l, self.d_l, self.total_h_dim,self.speaker_size_l)
        self.lsthm_a = LSTHM1(self.dh_a, self.d_a, self.total_h_dim,self.speaker_size_a)
        self.w=nn.Parameter(torch.ones(2))

        self.att = nn.Sequential(nn.Linear(self.total_h_dim, self.num_atts * self.total_h_dim))
        self.att_l=nn.Sequential(nn.Linear(self.dh_l,self.num_atts*self.dh_l),nn.ReLU(),nn.Dropout(map_dropout))
        self.att_a=nn.Sequential(nn.Linear(self.dh_a,self.num_atts*self.dh_a),nn.ReLU(),nn.Dropout(map_dropout))

        # self.reduce_dim_nn_l = nn.Sequential(nn.Linear(self.num_atts * self.dh_l, self.l_reduce_dim))
        # self.reduce_dim_nn_a = nn.Sequential(nn.Linear(self.num_atts * self.dh_a, self.a_reduce_dim))
        # self.reduce_dim_nn_v = nn.Sequential(nn.Linear(self.num_atts * self.dh_v, self.v_reduce_dim))
        # self.reduce_dim_nn = nn.Sequential(nn.Linear(self.num_atts * self.total_h_dim, self.total_reduce_dim))
        self.att_cross_modal=nn.Sequential(nn.Linear(self.total_h_dim, self.total_reduce_dim),nn.ReLU(),nn.Dropout(map_dropout))
        self.fc_z=nn.Sequential(nn.Linear(4*(self.total_h_dim+self.total_reduce_dim), self.total_h_dim),nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(self.total_reduce_dim, map_h), nn.ReLU(), nn.Dropout(map_dropout), nn.Linear(map_h, self.total_h_dim))

        self.nn_out = nn.Sequential(nn.Linear(final_out, h_out), nn.ReLU(), nn.Dropout(out_dropout), nn.Linear(h_out, output_dim))
        self.g_attention=general_attention(self.total_h_dim,self.total_reduce_dim,map_h,map_dropout)
        d_inner = 50
        n_head = 8
        d_k = 40
        d_v = 40
        self.encoder_l = EncoderLayer(100, d_inner, n_head, d_k, d_v)
        self.encoder_a = EncoderLayer(100, d_inner, n_head, d_k, d_v)
        self.encoder_v = EncoderLayer(512, d_inner, n_head, d_k, d_v)
        
        self.S = nn.Linear(self.speaker_size_l, 4*self.dh_l)  

    def forward(self, x, qmask):
        # x: L, B, D
        x_l = x[:, :, :self.d_l].to(x.device).permute(1, 0, 2)
        x_a = x[:, :, self.d_l:self.d_l + self.d_a].to(x.device).permute(1, 0, 2)
        x_l, a_l = self.encoder_l(x_l)
        x_a, a_a = self.encoder_a(x_a)
        x_l = x_l.permute(1, 0, 2)
        x_a = x_a.permute(1, 0, 2)

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
  
        N = x.shape[1]
        T = x.shape[0]
        self.h_l = torch.zeros(N, self.dh_l).to(x.device)
        self.h_a = torch.zeros(N, self.dh_a).to(x.device)
        self.c_l = torch.zeros(N, self.dh_l).to(x.device)
        self.c_a = torch.zeros(N, self.dh_a).to(x.device)
        self.z_t = torch.zeros(N, self.total_h_dim).to(x.device)
        self.q = torch.zeros(qmask.size()[1], qmask.size()[2], self.speaker_size_l).to(x.device) # batch, party, D_speaker
        all_h_ls, all_h_as = [], []
        all_c_ls, all_c_as = [], []
        all_z_ts = []
        all_q_ls, all_q_as = [], []
        output=[]
        for i in range(T):
            new_q, q_affine = self.cal_q(self.q, qmask[i])    # N, 2, 4*dh
            # q_affine = 0
            # current time step
            new_c_l, new_h_l = self.lsthm_l(x_l[i], *(self.c_l, self.h_l, self.z_t, q_affine))
            new_c_a, new_h_a = self.lsthm_a(x_a[i], *(self.c_a, self.h_a, self.z_t, q_affine))

            new_cs = torch.cat([new_c_l, new_c_a], dim=1)
            attention = F.softmax(torch.cat(torch.chunk(self.att(new_cs), self.num_atts, dim=1), dim=0), dim=1)     # 4*N, total_h_dim
            attended = attention * new_cs.repeat(self.num_atts, 1)      # 4*N, total_h_dim
            attention_l= F.softmax(torch.cat(torch.chunk(self.att_l(new_c_l), self.num_atts, dim=1), dim=0), dim=1)     # 4*N, dh_l
            attention_a= F.softmax(torch.cat(torch.chunk(self.att_a(new_c_a), self.num_atts, dim=1), dim=0), dim=1)     # 4*N, dh_a
            attended_l= attention_l * new_c_l.repeat(self.num_atts, 1)
            attended_a= attention_a * new_c_a.repeat(self.num_atts, 1)
 
            cross_modal=self.att_cross_modal(torch.cat([attended_l, attended_a], dim=-1))  # N, total_reduce_dim

            z_temp=torch.cat([cross_modal,attended],dim=-1)
            self.z_t=w1*self.fc_z(torch.cat(torch.chunk(z_temp, self.num_atts, dim=0), dim=1))+w2*self.z_t

            self.q = new_q
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            # all_q_ls.append(self.q_l)
            # all_q_as.append(self.q_a)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_z_ts.append(self.z_t)
            all_hs=torch.cat([self.h_l, self.h_a, self.z_t], dim=1)
            out=self.nn_out(all_hs)
            output.append(out)

        # last_h_l = all_h_ls[-1]
        # last_h_a = all_h_as[-1]
        # last_z_t = all_z_ts[-1]
        # last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_z_t], dim=1)
        # output = self.nn_out(last_hs)
        output=torch.stack(output,dim=0).permute(1, 0, 2)    # S, B, D
        output = output.reshape(-1, output.size()[-1])

        return output, x_a, x_l

    def cal_q(self, qtm, qmask):
        qm_idx = torch.argmax(qmask, 1) #本batch各对话当前说话人

        q0_sel = self._select_parties(qtm, qm_idx)  # N, 4*dim
        q_affine=self.S(q0_sel)     # N, 4*dim
    
        q_s=q_affine.unsqueeze(1).expand(-1,qmask.size()[1],-1)  # N, 2, 4*dim
 
        q_0= q0_sel.unsqueeze(1).expand(-1,qmask.size()[1],-1)  # N, 2, 4*dim
        qmask_ = qmask.unsqueeze(2) #qmask -> batch, party      N, 2, 1
 
        q_t = q_0*(1-qmask_) + q_s*qmask_      # N, 2, 4*dim
        
        return q_t, q_affine

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        # print(f'q0_sel:{q0_sel}')
        return q0_sel