import torch
from InstrumentsDS import InstrumentsDS
from torch.utils.data import random_split
from pandas import read_csv

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else :
    print("pas cuda :(")

ds = InstrumentsDS('data', 12)
ds.__getitem__(9)

num_items = len(ds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(ds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

n = 5
df = read_csv("data/metadata.csv")
print(df.groupby('classID').sample(n, random_state=12))