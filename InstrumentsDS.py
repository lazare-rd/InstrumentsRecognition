from torch.utils.data import DataLoader, IterableDataset, random_split
import pandas as pd
from AudioUtil import *
from datasets import load_dataset

# ----------------------------
# Instruments Dataset
# ----------------------------
class InstrumentsDS(IterableDataset):
  
    def __init__(self, start, end):
        super(InstrumentsDS).__init__()
        self.dataset = load_dataset("lazarerd/ClassicInstruments", streaming=True)
        self.start = start
        self.end = end
        
    def __iter__(self, idx):
        return map(self.convert_to_spec, self.dataset)

  
    def convert_to_spec(self, audio):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.mapping.iloc[idx, 0]
        # Get the Class ID
        class_id = self.mapping.iloc[idx, 1]

        aud = AudioUtil.open(audio_file)

        sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id