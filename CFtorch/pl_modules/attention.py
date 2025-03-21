import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_heads: int,
                 key_size: int,
                 value_size: int,
                 dropout_rate,):
        super(MultiHeadAttention, self).__init__()
        # assert input_dim % num_heads == 0, "Input dimension must be divisible by number of heads"
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.dropout_rate = dropout_rate
        # self.head_dim = input_dim // num_heads

        self.query_linear = nn.Linear(input_dim, key_size * num_heads)
        self.key_linear = nn.Linear(input_dim, key_size * num_heads)
        self.value_linear = nn.Linear(input_dim, value_size * num_heads)
        self.output_linear = nn.Linear(value_size * num_heads, input_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def split_heads(self, x, batch_size, size):
        x = x.view(batch_size, -1, self.num_heads, size)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations for query, key, and value
        Q = self.query_linear(query)
        # print('key shape',key.shape)
        # print('input dim', self.input_dim)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # Split the heads
        Q = self.split_heads(Q, batch_size, self.key_size)
        K = self.split_heads(K, batch_size, self.key_size)
        V = self.split_heads(V, batch_size, self.value_size)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.key_size, dtype=torch.float32))

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)  # Masked positions are assigned -inf

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and apply final linear transformation
        attention_output = attention_output.permute(0, 2, 1, 3)
        attention_output = attention_output.contiguous().view(batch_size, attention_weights.size()[-1], -1)
        output = self.output_linear(attention_output)

        return output, attention_weights


class CrystalFormer_lays(nn.Module):
    def __init__(self,
                 n_lay: int,
                 input_dim: int,
                 num_heads: int,
                 key_size: int,
                 value_size: int,
                 widering_facter: int,
                 dropout_rate,):
        super(CrystalFormer_lays, self).__init__()
        self.n_lay = n_lay
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.dropout_rate = dropout_rate

        atten_lays = []
        linea_lays1 = []
        linea_lays2 = []
        lay_norm1 = []
        lay_norm2 = []
        dropout1 = []
        dropout2 = []
        for _ in range(n_lay):
            atten_lays.append(
                MultiHeadAttention(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    key_size=key_size,
                    value_size=value_size,
                    dropout_rate=dropout_rate,
                )
            )

            linea_lays1.append(
                nn.Sequential(*[nn.Linear(input_dim, widering_facter * input_dim),
                                nn.GELU(),
                                nn.Linear(widering_facter * input_dim, input_dim)])
            )

            lay_norm1.append(nn.LayerNorm([input_dim]))
            lay_norm2.append(nn.LayerNorm([input_dim]))

            dropout1.append(nn.Dropout(p=dropout_rate))
            dropout2.append(nn.Dropout(p=dropout_rate))

        self.atten_lays = nn.ModuleList(atten_lays)
        self.linea_lays1 = nn.ModuleList(linea_lays1)
        self.lay_norm1 = nn.ModuleList(lay_norm1)
        self.lay_norm2 = nn.ModuleList(lay_norm2)
        self.dropout1 = nn.ModuleList(dropout1)
        self.dropout2 = nn.ModuleList(dropout2)

    def forward(self, h, mask=None):
        for i in range(self.n_lay):
            h_norm = self.lay_norm1[i](h)
            if mask == None:
                mask = torch.ones(h.size()[1], h.size()[1], device=self.device)
            h_attn, _ = self.atten_lays[i](h_norm, h_norm, h_norm, mask)  # should had mask
            h_attn = self.dropout1[i](h_attn)
            h = h + h_attn
            h_norm = self.lay_norm2[i](h)
            # print('h_norm shape', h_norm.shape)
            # print('self.input_dim', self.input_dim)
            h_dense = self.linea_lays1[i](h_norm)
            h_dense = self.dropout2[i](h_dense)
            h = h + h_dense

        return h
