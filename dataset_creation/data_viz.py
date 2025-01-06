from dataset_creation.AudioUtil import *
import torch

(sigP, srP) = AudioUtil.open("data/wav/piano/piano_1003.wav")
(sigG, srG) = AudioUtil.open("data/wav/guitar/guitar_1009.wav")
(sigV, srV) = AudioUtil.open("data/wav/violin/violin_1004.wav")
(sigC, srC) = AudioUtil.open("data/wav/cello/cello_1003.wav")

print(f"piano shape : {sigP.shape}\nsampling rate : {srP}\n")
print(f"guitar shape : {sigG.shape}\nsampling rate : {srG}\n")
print(f"violin shape : {sigV.shape}\nsampling rate : {srV}\n")
print(f"cello shape : {sigC.shape}\nsampling rate : {srC}\n")

specP = AudioUtil.spectro_gram((sigP, srP))
print(specP.shape)

AudioUtil.plot_spec(specP)