import os
import torch
import numpy as np

from audio import AudioPlayer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_data_str_int(data, thresholds=None):
    data = data.split(',')

    if thresholds is not None:
        flex_data = [round(apply_threshold(float(data[i]), 2200, 4100), 2) for i in range(5)]
    else:
        flex_data = data[:5]

    data = flex_data + data[8:-1] + [data[-1].strip()]
    data = [float(x) for x in data]
    data = ', '.join([str(x) for x in data])
    return data

def convert_data_str_torch(data, threshold=False, seq_len=20, id=0):
    for i in range(0, len(data)):
        data[i] = [float(d) for d in data[i].split(', ')]

    #print(len(data))
    #print(data)
    data_np = np.array(data[id:id + seq_len]).reshape((1, 11, 20))
    data_torch = torch.from_numpy(data_np).float()
    return data_torch

def apply_threshold(val, bottom=0, top=4000):
    return (val - bottom)/(top - bottom)

def save_data(data, filepath='test.txt'):
    with open(filepath, 'a') as f:
        f.write(data + '\n')

def read_data(filepath):
    with open(filepath, 'r') as f:
        data = f.read().splitlines()
        return data

def read_config(config_path):
    with open(os.path.join('config', f'{config_path}.txt'), 'r') as f:
        data = [float(x.split(':')[1].split(',')[0].strip()) for x in f.read().splitlines()]
        return data

def output_audio(predict:str, audio_folder:str = f"alphabet_audio/wav/"):
    player = AudioPlayer()
    
    filename = f'{predict}.wav'
    filepath = f'{audio_folder}{filename}'
    
    player.play_audio(filepath)
    del(player)

if __name__ == '__main__':
    data = read_data('./data/a/a-1.txt')
    data = [convert_data_str_int(x) for x in data]
    print(data)
