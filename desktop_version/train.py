from TrainingUtil import Training
from models.CNN_Classifier import CNN_Classifier
from InstrumentsDS import InstrumentsDS
from torch.utils.data import random_split, Subset
import torch
import json

NUM_EPOCH = 10
DATASET_SIZE = 4000
DATASET_HZ = "spec"

if __name__ == "__main__":

    model = CNN_Classifier()


    ds = InstrumentsDS(f'data/{DATASET_HZ}', DATASET_SIZE)

    num_items = len(ds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(ds, [num_train, num_val])
    print(num_items)

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)


    loss_over_epochs, acc_over_epochs, eval_loss_over_epochs, eval_acc_over_epochs = Training.training_loop(model, train_dl, val_dl, NUM_EPOCH)
    to_json = {
        "loss": loss_over_epochs,
        "accuracy": acc_over_epochs,
        "eval_loss": eval_loss_over_epochs,
        "eval_accuracy": eval_acc_over_epochs
    }

    torch.save(model.state_dict(), f'models/saved_models/CNN_{DATASET_SIZE}_{DATASET_HZ}Hz.pth')
    with open(f'models/training_results/training_results_CNN_{DATASET_SIZE}_{DATASET_HZ}Hz.json', 'w') as json_file:
        json.dump(to_json, json_file)

    print("Model saved")