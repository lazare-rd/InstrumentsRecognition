from AudioUtil import AudioUtil
import os
import torchaudio
from tqdm import tqdm

audio_dir = ["data/48000/wav/cello", "data/48000/wav/guitar", "data/48000/wav/piano", "data/48000/wav/violin"]
audio_files =["cello", "guitar", "piano", "violin"]

def resample_loop(audio_dir, output_dir, target_sr):
  os.makedirs(output_dir, exist_ok=True)
  for file in tqdm(os.listdir(audio_dir)):
    if (file.endswith(".wav")):
      aud = AudioUtil.open(audio_dir + "/" + file)
      aud = AudioUtil.resample(aud, target_sr)
      torchaudio.save(output_dir + "/" + file, aud[0], target_sr, format="wav")
  print(f"Ressampled {audio_dir} to {target_sr}Hz")


def resample_all(audio_dir, target_sr):
  for i in range(len(audio_dir)):
    resample_loop(audio_dir[i], f"data/{target_sr}/wav/{audio_files[i]}", target_sr)

if __name__ == "__main__":
  resample_all(audio_dir, 16000)
  resample_all(audio_dir, 32000)
  resample_all(audio_dir, 8000)