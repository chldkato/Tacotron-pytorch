import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import tensorflow as tf
import numpy as np
matplotlib.use('Agg')


font_name = fm.FontProperties(fname="util/malgun.ttf").get_name()
matplotlib.rc('font', family=font_name, size=14)


def plot_alignment(alignment, path, text):
    text = text.rstrip('_').rstrip('~')
    alignment = alignment[:len(text)]
    _, ax = plt.subplots(figsize=(len(text)/3, 5))
    ax.imshow(np.transpose(alignment), aspect='auto', origin='lower')
    plt.xlabel('Encoder timestep')
    plt.ylabel('Decoder timestep')
    text = [x if x != ' ' else '' for x in list(text)]
    plt.xticks(range(len(text)), text)
    plt.tight_layout()
    plt.savefig(path, format='png')
