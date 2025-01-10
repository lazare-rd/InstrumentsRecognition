import json
import matplotlib.pyplot as plt

# Load the JSON file
with open('training_results/training_results_LSTM_alldataset.json', 'r') as f:
    data = json.load(f)

# Extract the accuracy values
train_accuracy = data['loss']
eval_accuracy = data['eval_loss']

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.plot(train_accuracy, label='Train Loss', marker='o')
plt.plot(eval_accuracy, label='Eval Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss vs Eval Loss - LSTM')
plt.legend()
plt.grid(True)
plt.show()