import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import EncoderLayer

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
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

        sums = input_affine + output_affine + hybrid_affine + speaker_affine    # N, 4*dim
        
        # biases are already part of W and U and V
        f_t = torch.sigmoid(sums[:, :self.cell_size])   # N, 128
        i_t = torch.sigmoid(sums[:, self.cell_size:2*self.cell_size])   # N, 128
        o_t = torch.sigmoid(sums[:, 2*self.cell_size:3*self.cell_size])     # N, 128
        ch_t = torch.tanh(sums[:, 3*self.cell_size:])       # N, 128
        c_t = f_t * ctm + i_t * ch_t
        h_t = torch.tanh(c_t) * o_t

        return c_t, h_t

class general_attention(nn.Module):
    def __init__(self, D_cs,D_x,hidden_size,dropout):
        super(general_attention,self).__init__()
        self.D_cs = D_cs
        self.D_x = D_x
        self.hidden_size=hidden_size
        self.dropout=dropout
        self.trans = nn.Sequential(nn.Linear(self.D_x,self.hidden_size), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(self.hidden_size, 
        self.D_cs))
        
    
    def forward(self, new_cs, x):
        x = self.trans(x)
        alpha = torch.mul(new_cs,x)
        alpha = F.softmax(alpha,dim=-1)
        alpha_sum = torch.sum(alpha, dim=-1, keepdim=True) 
        alpha = alpha/alpha_sum 
        
        attn=torch.mul(alpha,x)
        return attn



class MARN1(nn.Module):
    def __init__(self):
        super(MARN1, self).__init__()
        [self.d_l, self.d_a, self.d_v] = [100,100,512]
        [self.dh_l, self.dh_a, self.dh_v] = [128,16,64]
        [self.l_reduce_dim, self.a_reduce_dim, self.v_reduce_dim] = [16,128,100]
        self.total_h_dim = self.dh_l + self.dh_a + self.dh_v
        self.total_reduce_dim = self.l_reduce_dim + self.a_reduce_dim + self.v_reduce_dim
        self.speaker_size_l=4*self.dh_l
        self.speaker_size_a=4*self.dh_a
        self.speaker_size_v=4*self.dh_v
        self.num_atts = 4
        output_dim = 6
        # output_dim = 3
        final_out = 2 * self.total_h_dim
        h_out = 32
        out_dropout = 0.0
        map_h = 64
        map_dropout =0.3
        self.lsthm_l = LSTHM1(self.dh_l, self.d_l, self.total_h_dim,self.speaker_size_l)
        self.lsthm_a = LSTHM1(self.dh_a, self.d_a, self.total_h_dim,self.speaker_size_a)
        self.lsthm_v = LSTHM1(self.dh_v, self.d_v, self.total_h_dim,self.speaker_size_v)
        # self.w=nn.Parameter(torch.ones(4))

        self.att = nn.Sequential(nn.Linear(self.total_h_dim, self.num_atts * self.total_h_dim))

        self.reduce_dim_nn_l = nn.Sequential(nn.Linear(self.num_atts * self.dh_l, self.l_reduce_dim))
        self.reduce_dim_nn_a = nn.Sequential(nn.Linear(self.num_atts * self.dh_a, self.a_reduce_dim))
        self.reduce_dim_nn_v = nn.Sequential(nn.Linear(self.num_atts * self.dh_v, self.v_reduce_dim))

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
        

    def forward(self, x,qmask):
        x_l = x[:, :, :self.d_l]
        x_a = x[:, :, self.d_l:self.d_l + self.d_a]
        x_v = x[:, :, self.d_l + self.d_a:]
        x_l, a_l = self.encoder_l(x_l)
        x_a, a_a = self.encoder_a(x_a)
        x_v, a_v = self.encoder_v(x_v)
        # x is T*N*d
        # print(qmask.shape)
        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        # w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))
        N = x.shape[1]
        T = x.shape[0]
        self.h_l = torch.zeros(N, self.dh_l).to(x.device)
        self.h_a = torch.zeros(N, self.dh_a).to(x.device)
        self.h_v = torch.zeros(N, self.dh_v).to(x.device)
        self.c_l = torch.zeros(N, self.dh_l).to(x.device)
        self.c_a = torch.zeros(N, self.dh_a).to(x.device)
        self.c_v = torch.zeros(N, self.dh_v).to(x.device)
        self.z_t = torch.zeros(N, self.total_h_dim).to(x.device)
        self.q_l = torch.zeros(qmask.size()[1], qmask.size()[2], self.speaker_size_l).to(x.device) # batch, party, D_speaker
        self.q_a = torch.zeros(qmask.size()[1], qmask.size()[2], self.speaker_size_a).to(x.device) # batch, party, D_speaker
        self.q_v = torch.zeros(qmask.size()[1], qmask.size()[2], self.speaker_size_v).to(x.device) # batch, party, D_speaker
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_z_ts = []
        all_q_ls = []
        all_q_as = []
        all_q_vs = []
        output=[]
        for i in range(T):
            # current time step
            new_c_l, new_h_l, new_q_l  = self.lsthm_l(x_l[i], *(self.c_l, self.h_l, self.z_t,self.q_l,qmask[i]))
            new_c_a, new_h_a ,new_q_a = self.lsthm_a(x_a[i], *(self.c_a, self.h_a, self.z_t,self.q_a,qmask[i]))
            new_c_v, new_h_v ,new_q_v = self.lsthm_v(x_v[i], *(self.c_v, self.h_v, self.z_t,self.q_v,qmask[i]))

            new_cs = torch.cat([new_c_l, new_c_a, new_c_v], dim=1)
            attention = F.softmax(torch.cat(torch.chunk(self.att(new_cs), self.num_atts, dim=1), dim=0), dim=1)
            attended = attention * new_cs.repeat(self.num_atts, 1)
            reduce_l = self.reduce_dim_nn_l(torch.cat(torch.chunk(attended[:, :self.dh_l], self.num_atts, dim=0), dim=1))
            reduce_a = self.reduce_dim_nn_a(torch.cat(torch.chunk(attended[:, self.dh_l:self.dh_l+self.dh_a], self.num_atts, dim=0), dim=1))
            reduce_v = self.reduce_dim_nn_v(torch.cat(torch.chunk(attended[:, self.dh_l+self.dh_a:], self.num_atts, dim=0), dim=1))
            # self.z_t = self.fc(torch.cat([reduce_l, reduce_a, reduce_v], dim=1))
            self.z_t = self.g_attention(new_cs,torch.cat([reduce_l, reduce_a, reduce_v], dim=1))
            self.h_l, self.c_l, self.q_l = new_h_l, new_c_l, new_q_l
            self.h_a, self.c_a, self.q_a= new_h_a, new_c_a ,new_q_a
            self.h_v, self.c_v, self.q_v = new_h_v, new_c_v ,new_q_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_q_ls.append(self.q_l)
            all_q_as.append(self.q_a)
            all_q_vs.append(self.q_v)
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
        output=torch.cat(output,dim=0)
        return output



class MARN_early(nn.Module):
    def __init__(self):
        super(MARN_early, self).__init__()
        self.d_h=128+16+64
        self.speaker_size = 4*self.d_h
        self.d_in=100+100+512
        [self.d_l, self.d_a, self.d_v] = [100,100,512]
        [self.dh_l, self.dh_a, self.dh_v] = [128,16,64]
        [self.l_reduce_dim, self.a_reduce_dim, self.v_reduce_dim] = [16,128,100]
        self.total_h_dim = self.dh_l + self.dh_a + self.dh_v
        self.total_reduce_dim = self.l_reduce_dim + self.a_reduce_dim + self.v_reduce_dim
        self.speaker_size_l=4*self.dh_l
        self.speaker_size_a=4*self.dh_a
        self.speaker_size_v=4*self.dh_v
        self.num_atts = 4
        output_dim = 6
        # output_dim = 3
        final_out = 2 * self.total_h_dim
        h_out = 32
        out_dropout = 0.0
        map_h = 64
        map_dropout =0.3
        self.lsthm = LSTHM1(self.d_h,self.d_in,self.total_h_dim,self.speaker_size)
     
        self.att = nn.Sequential(nn.Linear(self.total_h_dim, self.num_atts * self.total_h_dim))

       
        self.reduce_dim_nn = nn.Sequential(nn.Linear(self.num_atts*self.d_h,self.total_reduce_dim))


        self.fc = nn.Sequential(nn.Linear(self.total_reduce_dim, map_h), nn.ReLU(), nn.Dropout(map_dropout), nn.Linear(map_h, self.total_h_dim))

        self.nn_out = nn.Sequential(nn.Linear(final_out, h_out), nn.ReLU(), nn.Dropout(out_dropout), nn.Linear(h_out, output_dim))
        self.g_attention=general_attention(self.total_h_dim,self.total_reduce_dim,map_h,map_dropout)
        self.encoder=EncoderLayer(712,50,8,40,40)
       

    def forward(self, x,qmask):
        N = x.shape[1]
        T = x.shape[0]
        x,a=self.encoder(x)
       
        self.h = torch.zeros(N, self.d_h).to(x.device)
        self.c = torch.zeros(N, self.d_h).to(x.device)
       
        self.z_t = torch.zeros(N, self.total_h_dim).to(x.device)
        self.q = torch.zeros(qmask.size()[1], qmask.size()[2], self.speaker_size).to(x.device) # batch, party, D_speaker
       
        all_h = []
        all_c = []
        all_z = []
        all_q = []
       
        output=[]
        for i in range(T):
            # current time step
            new_c ,new_h, new_q= self.lsthm(x[i],*(self.c, self.h, self.z_t,self.q,qmask[i]))
           
            new_cs = new_c
            attention = F.softmax(torch.cat(torch.chunk(self.att(new_cs), self.num_atts, dim=1), dim=0), dim=1)
            attended = attention * new_cs.repeat(self.num_atts, 1)
           
            reduced=self.reduce_dim_nn(torch.cat(torch.chunk(attended, self.num_atts, dim=0), dim=1))
            self.z_t = self.g_attention(new_cs,reduced)
            self.h,self.c,self.q = new_h, new_c, new_q
            all_h.append(self.h)
            all_c.append(self.c)
            all_q.append(self.q)
            all_z.append(self.z_t)
           
            all_hs=torch.cat([self.h, self.z_t], dim=1)
            out=self.nn_out(all_hs)
            # print(out.shape)
            output.append(out)

      
        output=torch.cat(output,dim=0)
        return output



if __name__ == '__main__':
    t=torch.randn((44,30,712)).cuda()
    # print(t.is_cuda)
    USE_CUDA = torch.cuda.is_available() 
    if USE_CUDA:
       model = MARN1().cuda()
    out=model(t)
    print(out.shape)
