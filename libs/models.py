import copy
import torch
from torch import nn
from libs import *



class Transformer(nn.Module):
    def __init__(self, n_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.5, ffn_rate=0.5, eps=1e-6):
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(n_layer=n_layers, d_model=d_model, num_heads=num_heads, dff=dff,
         input_vocab_size=input_vocab_size, maximum_position_encoding=pe_input, rate=rate, ffn_rate=ffn_rate, eps=eps)
        self.decoder = Decoder(n_layer=n_layers, d_model=d_model, num_heads=num_heads, dff=dff,
         target_vocab_size=target_vocab_size, maximum_position_encoding=pe_target, rate=rate, ffn_rate=ffn_rate, eps=eps)
        self.finer_layer = nn.Linear(d_model, target_vocab_size)
    
    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.tokenizer(inp, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar,enc_output, look_ahead_mask, dec_padding_mask)
        final_output =self.finer_layer(dec_output)
        return final_output, attention_weights


class Encoder(nn.Module):
    def __init__(self, n_layer, d_model, num_heads, dff,
     input_vocab_size, maximum_position_encoding, rate=0.1, ffn_rate=0.1, eps=1e-6):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.n_layers = n_layer

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_embed=d_model, d_model=d_model, num_heads=num_heads, dff=dff, rate=rate, ffn_rate=ffn_rate, eps=eps) for _ in range(self.n_layers)]
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x, mask=None):
        seq_len = x.size()[1]
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for layer in self.enc_layers:
            x = layer(x, mask)
            
        return x

class Decoder(nn.Module):
    def __init__(self, n_layer, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.5, ffn_rate=0.5, eps=1e-6):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.n_layer = n_layer
        self.embedding = nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_embed=d_model, d_model=d_model, num_heads=num_heads, dff=dff, rate=rate, ffn_rate=ffn_rate, eps=eps) for _ in range(self.n_layer)]
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.size()[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)
        
        for i, layer in enumerate(self.dec_layers):
            x, block1, block2 = layer(x, enc_output, look_ahead_mask, padding_mask)

            attention_weights[f"decoder_layer{i}_block1"] = block1
            attention_weights[f"decoder_layer{i}_block2"] = block2
        
        return x, attention_weights


if __name__=="__main__":
    sample_encoder = Encoder(n_layer=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500, maximum_position_encoding=10000)
    temp_input = torch.randint(0, 100, (64, 62), dtype= torch.int32)
    sample_encoder_output =sample_encoder(temp_input, mask=None)
    print(f"Encoder Output Shape: {sample_encoder_output.size()}")

    sample_decoder = Decoder(n_layer=2, d_model=512, num_heads=8, dff=2048, target_vocab_size=8000, maximum_position_encoding=5000)
    temp_input = torch.randint(0, 100, (64, 26), dtype= torch.int32)
    sample_decoder_output, attn =sample_decoder(temp_input, sample_encoder_output, look_ahead_mask = None, padding_mask=None)
    print(f"Decoder Output Shape: {sample_decoder_output.shape}")
    print(f"Decoder Attention Shape: {attn['decoder_layer1_block2'].shape}")

    sample_transformer = Transformer(n_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500, target_vocab_size=8000, pe_input=10000, pe_target=6000)
    temp_input = torch.randint(0, 100, (64,48), dtype=torch.int64)
    temp_target = torch.randint(0,100, (64,36), dtype=torch.int64)
    fn_out, _ = sample_transformer(temp_input, temp_target, None, None, None)
    print(f"Transformer Output Shape: {fn_out.size()}")

