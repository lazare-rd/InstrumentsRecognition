from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from dataset_creation.AudioUtil import *


# ----------------------------
# Instruments Dataset
# ----------------------------
class InstrumentsDS(Dataset):
    
    MIN_NUM_SAMPLES = 4395

    def __init__(self, data_dir, num_samples_per_class):
        super(InstrumentsDS).__init__()
        self.data_dir = data_dir
        self.csv = pd.read_csv("data/metadata.csv")
        if num_samples_per_class is None:
            self.mapping = self.csv
        else :
            if num_samples_per_class <= InstrumentsDS.MIN_NUM_SAMPLES:
                self.mapping = self.csv.groupby('classID').sample(num_samples_per_class, random_state=12)
            else:
                raise Exception(f"num_samples_per_class must be inferior to {InstrumentsDS.MIN_NUM_SAMPLES} or None")
        
    def __len__(self):
        return self.mapping.shape[0]
        
    def __getitem__(self, idx):
        if self.data_dir == "data/spec":
            tensor_file = f"{self.data_dir}/{self.mapping.iloc[idx, 0][:-4]}.pt"
            class_id = self.mapping.iloc[idx, 1]
            class_id = int(class_id)

            tensor = torch.load(tensor_file)

            return tensor, class_id
        
        else:
            audio_file = f"{self.data_dir}/{self.mapping.iloc[idx, 0]}"
            class_id = self.mapping.iloc[idx, 1]
            class_id = int(class_id)
            aud = AudioUtil.open(audio_file)

            sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)
            aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

            return aug_sgram, class_id