import torch, librosa
import numpy as np
from torch.nn import Module, Linear, ReLU, Dropout, Conv1d, ModuleList, BatchNorm1d, GRU, MaxPool1d, Sigmoid, Softmax, Tanh
from util.hparams import *
from copy import deepcopy


class prenet(Module):
    def __init__(self, input_dim):
        super(prenet, self).__init__()
        self.fc1 = Linear(input_dim, embedding_dim)
        self.fc2 = Linear(embedding_dim, encoder_dim)
        
    def forward(self, input_data, is_training):
        x = self.fc1(input_data)
        x = ReLU()(x)
        
        if is_training:
            x = Dropout()(x)
        
        x = self.fc2(x)
        x = ReLU()(x)
        
        if is_training:
            x = Dropout()(x)
            
        return x
    
    
class CBHG(Module):
    def __init__(self, K, conv_dim):
        super(CBHG, self).__init__()
        self.K = K
        self.conv_bank = ModuleList(
            [Conv1d(conv_dim[1], encoder_dim, kernel_size=k, padding=k//2) for k in range(1, self.K+1)])
        self.bn = BatchNorm1d(encoder_dim)
        
        self.conv1 = Conv1d(encoder_dim * K, conv_dim[0], kernel_size=3, padding=1)
        self.bn1 = BatchNorm1d(conv_dim[0])
        self.conv2 = Conv1d(conv_dim[0], conv_dim[1], kernel_size=3, padding=1)
        self.bn2 = BatchNorm1d(conv_dim[1])
        
        self.fc = Linear(conv_dim[1], encoder_dim)
        self.H = Linear(encoder_dim, encoder_dim)
        self.T = Linear(encoder_dim, encoder_dim)
        self.T.bias.data.fill_(-1)
        
        self.gru = GRU(encoder_dim, encoder_dim, batch_first=True, bidirectional=True)
        
    def forward(self, input_data, sequence_length):
        maxT = input_data.shape[-1]
        x = torch.cat([ReLU()(self.bn(conv(input_data)[:, :, :maxT])) for conv in self.conv_bank], dim=1)
        x = MaxPool1d(kernel_size=2, stride=1, padding=1)(x)[:, :, :maxT]
        x = ReLU()(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        highway_input = input_data + x
        highway_input = highway_input.transpose(1, 2)
        
        if self.K == 8:
            highway_input = self.fc(highway_input)

        for _ in range(4):
            H = self.H(highway_input)
            H = ReLU()(H)
            T = self.T(highway_input)
            T = Sigmoid()(T)
            highway_input = H * T + highway_input * (1.0 - T)
        
        x = highway_input
        if sequence_length is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, sequence_length, batch_first=True)
            
        x, _ = self.gru(x)
        
        if sequence_length is not None:
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x
    
    
class LuongAttention(Module):
    def __init__(self):
        super(LuongAttention, self).__init__()
        self.w = Linear(decoder_dim, decoder_dim)
        
    def forward(self, query, value):
        alignment = Softmax(dim=-1)(torch.matmul(query, self.w(value).transpose(1, 2)))
        context = torch.matmul(alignment, value)
        context = torch.cat([context, query], axis=-1)
        alignment = alignment.transpose(1, 2)
        return context, alignment
    
    
class BahdanauAttention(Module):
    def __init__(self):
        super(BahdanauAttention, self).__init__()
        self.w1 = Linear(decoder_dim, decoder_dim)
        self.w2 = Linear(decoder_dim, decoder_dim)
        
    def forward(self, query, value):
        q = torch.unsqueeze(self.w1(query), axis=2)
        v = torch.unsqueeze(self.w2(value), axis=1)
        score = torch.sum(Tanh()(q + v), dim=-1)
        alignment = Softmax()(score)
        context = torch.matmul(alignment, value)
        context = torch.cat([context, query], axis=-1)
        alignment = alignment.transpose(1, 2)
        return context, alignment
    
    
def griffin_lim(spectrogram):
    spec = deepcopy(spectrogram)
    for i in range(50):
        est_wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
        est_stft = librosa.stft(est_wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        phase = est_stft / np.maximum(1e-8, np.abs(est_stft))
        spec = spectrogram * phase
    wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
    return np.real(wav)
