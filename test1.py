import os, argparse, glob, torch
import numpy as np
from jamo import hangul_to_jamo
from models.tacotron import Tacotron
from util.text import text_to_sequence, sequence_to_text
from util.plot_alignment import plot_alignment
from util.hparams import *


sentences = [
  '정말로 사랑한담 기다려주세요'
]

checkpoint_dir = './checkpoint/1'
save_dir = './output'
os.makedirs(save_dir, exist_ok=True)


def inference(text, idx):
    seq = text_to_sequence(text)
    enc_input = torch.tensor(seq, dtype=torch.int64).unsqueeze(0)
    sequence_length = torch.tensor([len(seq)], dtype=torch.int32)
    dec_input = torch.from_numpy(np.zeros((1, mel_dim), dtype=np.float32))
    
    pred, alignment = model(enc_input, sequence_length, dec_input, is_training=False, mode='inference')
    pred = pred.squeeze().detach().numpy()
    alignment = np.squeeze(alignment.detach().numpy(), axis=0)

    np.save(os.path.join(save_dir, 'mel-{}'.format(idx)), pred, allow_pickle=False)

    input_seq = sequence_to_text(seq)
    alignment_dir = os.path.join(save_dir, 'align-{}.png'.format(idx))
    plot_alignment(alignment, alignment_dir, input_seq)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', default=None)
    args = parser.parse_args()
    
    model = Tacotron(K=16, conv_dim=[128, 128])
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['model'])
    
    for i, text in enumerate(sentences):
        jamo = ''.join(list(hangul_to_jamo(text)))
        inference(jamo, i)
