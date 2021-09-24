import os, argparse, traceback, glob, random, itertools, time, torch, threading, queue
import numpy as np
import torch.optim as optim
from models.tacotron import post_CBHG
from torch.nn import L1Loss
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from util.hparams import *


data_dir = './data'
mel_list = sorted(glob.glob(os.path.join(data_dir + '/mel', '*.npy')))
spec_list = sorted(glob.glob(os.path.join(data_dir + '/spec', '*.npy')))
mel_len = np.load(os.path.join(data_dir + '/mel_len.npy'))

def DataGenerator():
    while True:
        idx_list = np.random.choice(len(mel_list), batch_group, replace=False)
        idx_list = sorted(idx_list)
        idx_list = [idx_list[i : i + batch_size] for i in range(0, len(idx_list), batch_size)]
        random.shuffle(idx_list)

        for idx in idx_list:
            random.shuffle(idx)

            mel = [torch.from_numpy(np.load(mel_list[mel_len[i][1]])) for i in idx]
            spec = [torch.from_numpy(np.load(spec_list[mel_len[i][1]])) for i in idx]

            mel = pad_sequence(mel, batch_first=True)
            spec = pad_sequence(spec, batch_first=True)

            yield [mel, spec]
            
            
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

    model = post_CBHG(K=8, conv_dim=[256, mel_dim]).cuda()

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
                mel, target = train_loader.next()
                mel = mel.float().cuda()
                target = target.float().cuda()

                pred = model(mel)
                loss = L1Loss()(pred, target)

                model.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                print('step: {}, loss: {:.5f}, {:.3f} sec/step'.format(step, loss, time.time() - start))

                if step % checkpoint_step == 0:
                    save_dir = './ckpt/' + args.name + '/2'
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
    save_dir = os.path.join('./ckpt/' + args.name, '2')
    os.makedirs(save_dir, exist_ok=True)
    train(args)
