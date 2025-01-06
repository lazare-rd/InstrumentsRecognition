from datasets import load_dataset

meta = "data/metadata/metadata_ClassicInstruments.csv"
data_files = ['https://s3.amazonaws.com/datasets.huggingface.co/lazarerd/ClassicInstruments/tree/main/wav/cello/cello_1.wav', "https://huggingface.co/datasets/lazarerd/ClassicInstruments/tree/main/wav/cello/cello_2.wav"]
data_dir = "https://huggingface.co/datasets/lazarerd/ClassicInstruments/wav/cello"

ds = load_dataset("audiofolder", data_files="https://huggingface.co/datasets/lazarerd/ClassicInstruments/wav/cello/cello_1.wav", streaming=True)
print(type(next(iter(ds))))