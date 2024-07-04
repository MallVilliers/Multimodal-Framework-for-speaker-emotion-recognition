from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_model2, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # 三个线性层做矩阵乘法生成q, k, v.
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model2, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model2, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        # ScaledDotProductAttention见下方
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
 
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # b: batch_size, lq: translation task的seq长度, n: head数, dv: embedding vector length
        # Separate different heads: b x lq x n x dv. 
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # project & reshape
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting. 
            # (batchSize, 1, seqLen) -> (batchSize, 1, 1, seqLen)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # view只能用在 contiguous 的 variable 上
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # add & norm
        q += residual

        q = self.layer_norm(q)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q x k^T
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # 把mask中为0的数置为-1e9, 用于decoder中的masked self-attention
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # dim=-1表示对最后一维softmax
        # attn = self.dropout(F.softmax(attn, dim=-1))
        attn = F.softmax(attn, dim=-1)
        # attn = torch.where(attn < 0.2, torch.zeros(1).to(q.device), attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(d_in, 100)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        # add & norm
        x += residual

        x = self.layer_norm(x)
        # x = self.fc(x)

        return x


class EncoderLayer(nn.Module):
    ''' 
    Compose with two layers 

    d_model: 特征维度
    d_inner 
    
    '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


if __name__ == '__main__':
    d_model = 100
    d_inner = 50
    n_head = 8
    d_k = 40
    d_v = 40

    enc_input = torch.randn(32, 60, 100)
    m = EncoderLayer(d_model, d_inner, n_head, d_k, d_v)
    o = m(enc_input)
    print(1)
