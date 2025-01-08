from AudioUtil import AudioUtil
import os
import torch
from tqdm import tqdm

audio_dir = ["data/48000/wav/cello", "data/48000/wav/guitar", "data/48000/wav/piano", "data/48000/wav/violin"]
audio_files =["cello", "guitar", "piano", "violin"]

def gen_spec(audio_file):
    aud = AudioUtil.open(audio_file)
    sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    return aug_sgram

def gen_spec_for_dir(audio_dir, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  for file in tqdm(os.listdir(audio_dir)):
    if (file.endswith(".wav")):
      spec = gen_spec(f"{audio_dir}/{file}")
      torch.save(spec, f"{output_dir}/{file[:-4]}.pt")

def gen_spec_for_all(audio_dir):
  for i in range(len(audio_dir)):
    gen_spec_for_dir(audio_dir[i], f"data/spec/wav/{audio_files[i]}")

if __name__ == "__main__":
    gen_spec_for_all(audio_dir)