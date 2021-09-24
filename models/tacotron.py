import numpy as np
from torch.nn import Module, Embedding, Linear, GRUCell
from models.modules import *
from util.hparams import *


class Encoder(Module):
    def __init__(self, K, conv_dim):
        super(Encoder, self).__init__()
        self.embedding = Embedding(symbol_length, embedding_dim)
        self.prenet = prenet(embedding_dim)
        self.cbhg = CBHG(K, conv_dim)
        
    def forward(self, enc_input, sequence_length, is_training):
        x = self.embedding(enc_input)
        x = self.prenet(x, is_training=is_training)
        x = x.transpose(1, 2)
        x = self.cbhg(x, sequence_length)
        return x

    
class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.prenet = prenet(mel_dim)
        self.attention_rnn = GRUCell(encoder_dim, decoder_dim)
        self.attention = LuongAttention()
        self.proj1 = Linear(decoder_dim * 2, decoder_dim)
        self.dec_rnn1 = GRUCell(decoder_dim, decoder_dim)
        self.dec_rnn2 = GRUCell(decoder_dim, decoder_dim)
        self.proj2 = Linear(decoder_dim, mel_dim * reduction)
        
    def forward(self, batch, dec_input, enc_output, mode):
        if mode == 'train':
            dec_input = dec_input.transpose(0, 1)
            attn_rnn_state = torch.zeros(batch, decoder_dim).cuda()
            dec_rnn_state1 = torch.zeros(batch, decoder_dim).cuda()
            dec_rnn_state2 = torch.zeros(batch, decoder_dim).cuda()
        else:
            attn_rnn_state = torch.zeros(batch, decoder_dim)
            dec_rnn_state1 = torch.zeros(batch, decoder_dim)
            dec_rnn_state2 = torch.zeros(batch, decoder_dim)
        
        iters = dec_input.shape[0] if mode == 'train' else max_iter+1
        
        for i in range(iters):
            inp = dec_input[i] if mode == 'train' else dec_input
            x = self.prenet(inp, is_training=True)
            attn_rnn_state = self.attention_rnn(x, attn_rnn_state)
            attn_rnn_state = attn_rnn_state.unsqueeze(1)
            context, align = self.attention(attn_rnn_state, enc_output)

            dec_rnn_input = self.proj1(context)
            dec_rnn_input = dec_rnn_input.squeeze(1)

            dec_rnn_state1 = self.dec_rnn1(dec_rnn_input, dec_rnn_state1)
            dec_rnn_input = dec_rnn_input + dec_rnn_state1
            dec_rnn_state2 = self.dec_rnn2(dec_rnn_input, dec_rnn_state2)
            dec_rnn_output = dec_rnn_input + dec_rnn_state2

            dec_out = self.proj2(dec_rnn_output)

            dec_out = dec_out.unsqueeze(1)
            attn_rnn_state = attn_rnn_state.squeeze(1)

            if i == 0:
                mel_out = torch.reshape(dec_out, [batch, -1, mel_dim])
                alignment = align
            else:
                mel_out = torch.cat([mel_out, torch.reshape(dec_out, [batch, -1, mel_dim])], dim=1)
                alignment = torch.cat([alignment, align], dim=-1)
                
            if mode == 'inference':
                dec_input = mel_out[:, reduction * (i+1) - 1, :]

        return mel_out, alignment
    
    
class Tacotron(Module):
    def __init__(self, K, conv_dim):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(K, conv_dim)
        self.decoder = Decoder()
        
    def forward(self, enc_input, sequence_length, dec_input, is_training, mode):
        batch = dec_input.shape[0]
        x = self.encoder(enc_input, sequence_length, is_training)
        x = self.decoder(batch, dec_input, x, mode)
        return x
    

class post_CBHG(Module):
    def __init__(self, K, conv_dim):
        super(post_CBHG, self).__init__()
        self.cbhg = CBHG(K, conv_dim)
        self.fc = Linear(256, n_fft // 2 + 1)
        
    def forward(self, mel_input):
        x = self.cbhg(mel_input.transpose(1, 2), None)
        x = self.fc(x)
        return x
    