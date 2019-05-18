import subprocess, argparse
from scipy.io.wavfile import write
import os, shutil, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from predict_func import predict
import librosa
import librosa.display
from termcolor import colored
import time
import atexit

# Vars ###########
CH1_MAX = 270
CH2_MAX = 270
CH3_MAX = 220
CH4_MAX = 500
NB_CHANNELS = 4
CHANNELS = ['ch1','ch2','ch3','ch4']
STREAM_FILES = os.getcwd() + "/stream_files"
AUDIO = os.getcwd() + "/latest_wav/"
IMG = os.getcwd() + "/latest_spec/"

def latest_txt_files(path,qty):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files if basename.endswith(".txt")]
    s_paths = sorted(paths, key=os.path.getctime, reverse=True)
    ss_paths = sorted(s_paths[:qty], key=os.path.getctime)
    return ss_paths

def csv2wav(ch_name, ch_df, o_file_path):
    data = np.array(ch_df, dtype='float64')
    if ch_name == "ch1":
        data /= CH1_MAX
    elif ch_name == "ch2":
        data /= CH2_MAX
    elif ch_name == "ch3":
        data /= CH3_MAX
    elif ch_name == "ch4":
        data /= CH4_MAX
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # name = "%s/%s.wav"%(output_dir,ch)
    # wavfile.write(name, 200, data)
    data = signal.resample(data, 40 * 200 * 2)
    make_spectrograms(data, 8000, o_file_path)

def preprocess(samples, sample_rate, multiplier=1):
    sr = sample_rate * multiplier
    padded = np.zeros(sr)
    samples = samples[:sr]
    padded[:samples.shape[0]] = samples
    return padded

def make_spectrograms(samples, sample_rate, o_file_path): 
    changed = preprocess(samples, sample_rate, 1)
    S = librosa.feature.melspectrogram(changed, sr=sample_rate, n_mels=128, fmax=512)
    log_S = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(1.28, 1.28), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.axis('off')
    librosa.display.specshow(log_S)                          
    plt.savefig(o_file_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
        default="vocal", 
        help="The type of model, vocal for voice, subvocal for subvocal")
    args = parser.parse_args()

    while (True):
        #print(colored("-" * 21 + "\nSay command : \n" + "-" * 21, 'white'))
        fls=latest_txt_files(STREAM_FILES,1)
        print(fls[0])
        list_ = []
        #print(fls)
        for file_ in fls:
            df = pd.read_csv(file_,index_col=None, header=0)
            list_.append(df)
            
        frame = pd.concat(list_, axis = 0, ignore_index = True)
        # need to be replace with constants obtained from main data

        for channel in range(1, NB_CHANNELS + 1):
            ch_name = 'ch{}'.format(channel)
            ch_df = frame[ch_name] 
            o_file_path = os.path.join(IMG, ch_name, "00000.png") 
            csv2wav(ch_name, ch_df, o_file_path)

        predict(args.model)
        time.sleep(2.5)
        #process("downsampled","latest_spec", args.model)
        # pure_process("downsampled", "latest_spec", args.model)
        # shutil.rmtree(os.getcwd() + "/downsampled")



        # section = frame['ch4'][0]
        # if section > 100 or section < -100:
        #     print(colored(frame['ch4'][0], 'red'))
        # else:
        #     print(frame['ch4'][0])