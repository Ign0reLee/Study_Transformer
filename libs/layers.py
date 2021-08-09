import copy
import torch
import numpy as np
from torch import nn


class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_model, dff,  rate=0.5):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=rate)
        self.fc2 = nn.Linear(dff, d_model)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_embed, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.prob = nn.Softmax(dim=-1)

        self.wq = nn.Linear(d_embed, d_model)
        self.wk = nn.Linear(d_embed, d_model)
        self.wv = nn.Linear(d_embed, d_model)
        
        self.fc = nn.Linear(d_model, d_model)

    def split_head(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1,2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        query = self.wq(query)
        key   = self.wk(key)
        value = self.wv(value)

        query = self.split_head(query, batch_size)
        key   = self.split_head(key,   batch_size)
        value = self.split_head(value, batch_size)

        if mask is not None:
            mask = mask.unsqueeze(1)

        scaled_attetnion, attention_weights = self.calculate_attention(query, key, value, mask)
        out = scaled_attetnion.transpose(1,2)
        out = out.contiguous().view(batch_size, -1,  self.d_model)
        out = self.fc(out)
        return out, attention_weights


    def calculate_attention(self, value, key, query, mask=None):
        # Query Shape: (Batch, d_k, Squence_Length)
        # Key   Shape: (Batch, d_k, Squence_Length)
        # Value Shape: (Batch, d_k, Squence_Length)
        d_k = key.size(-1)
        attention_score = torch.matmul(query, key.transpose(-2, -1))
        attention_score = attention_score / np.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, 1e-9)
        attention_prob = self.prob(attention_score)
        out = torch.matmul(attention_prob, value)
        return out, attention_prob

class BiPartiteAttention(nn.Module):
    
    def __init__(self, d_model, d_embed, num_heads, duplex=False):
        super(BiPartiteAttention, self).__init__()

        self.gamma = nn.parameter.Parameter(torch.randint(size=(d_model, d_model)),requires_grad=True)
        self.beta  = nn.parameter.Parameter(torch.randint(size=(d_model, d_model)),requires_grad=True)
        self.mha = MultiHeadAttention(d_model=d_model, d_embed= d_embed, num_heads=num_heads)
        self.duplex = duplex
        self.norm = nn.LayerNorm()
    
    def w(self, x):
        mean = torch.mean(x)
        std  = torch.std(x)
        return (x - mean) / std
    
    def forward(self, query, key, value, mask=None):

        # query : X
        # key   : Y
        # value : Y

        attn = self.mha(query, key, value)
        u_a  = self.norm(query + attn)


class EncoderLayer(nn.Module):
    def __init__(self, d_embed, d_model, num_heads, dff, rate=0.1, ffn_rate=0.5, eps=1e-6):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_embed=d_embed, d_model=d_model, num_heads=num_heads)
        self.ffn = PositionWiseFeedForwardLayer(d_model=d_model, dff=dff, rate=ffn_rate)

        self.layernorm1 = nn.LayerNorm(d_model, eps=eps)
        self.layernorm2 = nn.LayerNorm(d_model, eps=eps)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class DecoderLayer(nn.Module):
    def __init__(self, d_embed, d_model, num_heads, dff, rate=0.1, ffn_rate=0.5, eps=1e-6):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model=d_model, d_embed=d_embed, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(d_model=d_model, d_embed=d_embed, num_heads=num_heads)

        self.ffn = PositionWiseFeedForwardLayer(d_model=d_model, dff=dff, rate=ffn_rate)

        self.layernorm1 = nn.LayerNorm(d_model, eps=eps)
        self.layernorm2 = nn.LayerNorm(d_model, eps=eps)
        self.layernorm3 = nn.LayerNorm(d_model, eps=eps)

        self.dropout1 = nn.Dropout(p=rate)
        self.dropout2 = nn.Dropout(p=rate)
        self.dropout3 = nn.Dropout(p=rate)
    
    def forward(self, x, enc_out, look_ahead_mask=None, padding_mask=None):

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 =self.mha2(enc_out, enc_out, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

        



if __name__ == "__main__":
    temp_mha = MultiHeadAttention(d_model=512, d_embed=512, num_heads=8)
    y = torch.rand((1,60,512))
    out, attn = temp_mha(y, y, y, mask=None)
    print(f"Multi-Head Attention Output Shape: {out.shape}")
    print(f"Multi-Head Attention Attention Score Shape: {attn.shape}")

    sample_encoder_layer = EncoderLayer(512,512, 8, 2048)
    sample_encoder_output = sample_encoder_layer(torch.rand((64,43,512)))
    print(f"Encoder Output : {sample_encoder_output.shape}")

    sample_decoder_layer = DecoderLayer(512, 512, 8 , 2048)
    sample_decoder_output, _, _ = sample_decoder_layer(torch.rand((64,26,512)), sample_encoder_output, None, None)
    print(f"Decoder Output : {sample_decoder_output.shape}")