import os, argparse, glob, torch, scipy, librosa
import numpy as np
import soundfile as sf
from models.tacotron import post_CBHG
from models.modules import griffin_lim
from util.hparams import *


checkpoint_dir = './checkpoint/2'
save_dir = './output'
os.makedirs(save_dir, exist_ok=True)
mel_list = glob.glob(os.path.join(save_dir, '*.npy'))


def inference(text, idx):
    mel = torch.from_numpy(text).unsqueeze(0)
    pred = model(mel)
    pred = pred.squeeze().detach().numpy() 
    pred = np.transpose(pred)
    
    pred = (np.clip(pred, 0, 1) * max_db) - max_db + ref_db
    pred = np.power(10.0, pred * 0.05)
    wav = griffin_lim(pred ** 1.5)
    wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)
    wav = librosa.effects.trim(wav, frame_length=win_length, hop_length=hop_length)[0]
    wav = wav.astype(np.float32)
    sf.write(os.path.join(save_dir, '{}.wav'.format(idx)), wav, sample_rate)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', default=None)
    args = parser.parse_args()
    
    model = post_CBHG(K=8, conv_dim=[256, mel_dim])
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['model'])
    
    for i, fn in enumerate(mel_list):
        mel = np.load(fn)
        inference(mel, i)
