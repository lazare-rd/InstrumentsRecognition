import torch
from torch import nn

class Training:

    CRITERION = nn.CrossEntropyLoss()

    def validation_loop(model, val_dl, criterion=CRITERION):
        model.eval()
        correct = 0
        total_pred = 0
        loss = 0

        with torch.no_grad():
            for data in val_dl:
                inputs, labels = data[0], data[1]
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)

                loss += criterion(outputs, labels).item()      
                correct += (pred == labels).sum().item()
                total_pred += pred.shape[0]

        return loss / len(val_dl), correct / total_pred
    
    def training_loop(model, train_dl, val_dl, num_epochs, criterion=CRITERION):
        # Loss Function, Optimizer and Scheduler
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')
        loss_over_epochs = []
        acc_over_epochs = []
        eval_loss_over_epochs = []
        eval_acc_over_epochs = []

        # Repeat for each epoch
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_prediction = 0
            total_prediction = 0

            # Repeat for each batch in the training set
            for i, data in enumerate(train_dl):
                
                inputs, labels = data[0], data[1]

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Keep stats for Loss and Accuracy
                running_loss += loss.item()

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs,1)
                # Count of predictions that matched the target label
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]

                if i % 10 == 0:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            
            eval_loss, eval_acc = Training.validation_loop(model, val_dl)
            eval_loss_over_epochs.append(eval_loss)
            eval_acc_over_epochs.append(eval_acc)

            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            loss_over_epochs.append(avg_loss)
            acc_over_epochs.append(acc)

            print(f'Epoch: {epoch}\nLoss: {avg_loss}, Accuracy: {acc}\nValidation Loss: {eval_loss}, Validation Accuracy: {eval_acc}')

        print('Finished Training')
        return loss_over_epochs, acc_over_epochs, eval_loss_over_epochs, eval_acc_over_epochs