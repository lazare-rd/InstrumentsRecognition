from pydub import AudioSegment
from tqdm import tqdm
import csv


CELLO = AudioSegment.from_wav("/Users/mac/Documents/AMIS/ML/data/CelloBach.wav") #2h33 55s to 2h31mn21s
end_cello = 12081 * 1000
CELLO = CELLO[55000:end_cello]

GUITAR = AudioSegment.from_wav("/Users/mac/Documents/AMIS/ML/data/GuitareFlamenco.wav") #1h28 20s to 1h28mn15s
GUITAR = GUITAR[20000:]

PIANO = AudioSegment.from_wav("/Users/mac/Documents/AMIS/ML/data/PianoBach.wav") #1h59 00 to 1h58mn54s
PIANO = PIANO[:-6000]

VIOLIN = AudioSegment.from_wav("/Users/mac/Documents/AMIS/ML/data/ViolonBach.wav") #1h13 from 28s to 1h13mn12s
VIOLIN = VIOLIN[28000:]


five_seconds = 5 * 1000
class_mapping = {0: "cello", 1 : "guitar", 2 : "piano", 3 : "violin"}
mapping = [("path/fileID", "classID", "class")]

def split_wav(wav, type):
    type_name = class_mapping[type]
    for i in tqdm(range(int(wav.duration_seconds - 5))):
        j = i * 1000
        split = wav[j:j+five_seconds]
        mapping.append((f"data/wav/{type_name}/{type_name}_{i}.wav", type, type_name))
        split.export(f"data/wav/{type_name}/{type_name}_{i}.wav", format ="wav", tags={'type' : type})


def add_to_csv(map_list, file_name):
    with open(file_name, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile, delimiter=',')
        for tuple in map_list:
            writter.writerow([tuple[0], tuple[1], tuple[2]])


if __name__ == "__main__":
    split_wav(CELLO, 0)
    split_wav(GUITAR, 1)
    split_wav(PIANO, 2)
    split_wav(VIOLIN, 3)

    add_to_csv(mapping , 'data/metadata/ClassicInstruments.csv')

