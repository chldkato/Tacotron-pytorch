import os, argparse, traceback, glob, random, itertools, time, torch, threading, queue
import numpy as np
import torch.optim as optim
from models.tacotron import Tacotron
from torch.nn import L1Loss
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from util.text import sequence_to_text
from util.plot_alignment import plot_alignment
from util.hparams import *


data_dir = './data'
text_list = sorted(glob.glob(os.path.join(data_dir + '/text', '*.npy')))
mel_list = sorted(glob.glob(os.path.join(data_dir + '/mel', '*.npy')))
dec_list = sorted(glob.glob(os.path.join(data_dir + '/dec', '*.npy')))

fn = os.path.join(data_dir + '/mel_len.npy')
if not os.path.isfile(fn):
    mel_len_list = []
    for i in range(len(mel_list)):
        mel_length = np.load(mel_list[i]).shape[0]
        mel_len_list.append([mel_length, i])
    mel_len = sorted(mel_len_list)
    np.save(os.path.join(data_dir + '/mel_len.npy'), np.array(mel_len))

text_len = np.load(os.path.join(data_dir + '/text_len.npy'))
mel_len = np.load(os.path.join(data_dir + '/mel_len.npy'))

    
def DataGenerator():
    while True:
        idx_list = np.random.choice(len(mel_list), batch_group, replace=False)
        idx_list = sorted(idx_list)
        idx_list = [idx_list[i : i + batch_size] for i in range(0, len(idx_list), batch_size)]
        random.shuffle(idx_list)

        for idx in idx_list:
            random.shuffle(idx)

            text = [torch.from_numpy(np.load(text_list[mel_len[i][1]])) for i in idx]
            dec = [torch.from_numpy(np.load(dec_list[mel_len[i][1]])) for i in idx]
            mel = [torch.from_numpy(np.load(mel_list[mel_len[i][1]])) for i in idx]

            text_length = torch.tensor([text_len[mel_len[i][1]] for i in idx], dtype=torch.int32)
            text_length, _ = text_length.sort(descending=True)

            text = pad_sequence(text, batch_first=True)
            dec = pad_sequence(dec, batch_first=True)
            mel = pad_sequence(mel, batch_first=True)

            yield [text, dec, mel, text_length]
            
            
class Generator(threading.Thread):
    def __init__(self, generator):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(8)
        self.generator = generator
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
             raise StopIteration
        return next_item


def train(args):
    train_loader = Generator(DataGenerator())

    model = Tacotron(K=16, conv_dim=[128, 128]).cuda()

    optimizer = optim.Adam(model.parameters())

    step, epochs = 0, 0
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['step'],
        step = step[0]
        epoch = ckpt['epoch']
        print('Load Status: Epoch %d, Step %d' % (epoch, step))

    torch.backends.cudnn.benchmark = True

    try:
        for epoch in itertools.count(epochs):
            for _ in range(batch_group):
                start = time.time()
                text, dec, target, text_length = train_loader.next()
                text = text.cuda()
                dec = dec.float().cuda()
                target = target.float().cuda()

                pred, alignment = model(text, text_length, dec, is_training=True, mode='train')
                loss = L1Loss()(pred, target)

                model.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                print('step: {}, loss: {:.5f}, {:.3f} sec/step'.format(step, loss, time.time() - start))

                if step % checkpoint_step == 0:
                    save_dir = './ckpt/' + args.name + '/1'
                    
                    input_seq = sequence_to_text(text[0].cpu().numpy())
                    input_seq = input_seq[:text_length[0].cpu().numpy()]
                    alignment_dir = os.path.join(save_dir, 'step-{}-align.png'.format(step))
                    plot_alignment(alignment[0].detach().cpu().numpy(), alignment_dir, input_seq)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'epoch': epoch
                    }, os.path.join(save_dir, 'ckpt-{}.pt'.format(step)))

    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', default=None)
    parser.add_argument('--name', '-n', required=True)
    args = parser.parse_args()
    save_dir = os.path.join('./ckpt/' + args.name, '1')
    os.makedirs(save_dir, exist_ok=True)
    train(args)
