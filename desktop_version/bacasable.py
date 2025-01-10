import torch
from InstrumentsDS import InstrumentsDS
from torch.utils.data import random_split
from pandas import read_csv
import json
from dataset_creation.AudioUtil import AudioUtil

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else :
    print("pas cuda :(")

ds = InstrumentsDS('data', 12)

num_items = len(ds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(ds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

n = 5
df = read_csv("data/metadata.csv")
print(df.iloc[9])
df_sample = df.groupby('classID').sample(n, random_state=12) 
df_sample.reset_index(drop=True)
print(df_sample.iloc[9])
to_json = {
    "loss": [0],
    "accuracy": [1, 2, 3],
    "eval_loss": [4, 5, 6],
    "eval_accuracy": [7, 8, 9]
}

audio_file = "data/48000/wav/cello/cello_1.wav"
aud = AudioUtil.open(audio_file)
sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=2048, hop_len=None)
aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
print(aug_sgram.shape)