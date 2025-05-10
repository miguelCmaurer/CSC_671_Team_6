from binary_model.binary_mlp import evaluate_model
from binary_model.usable_binary_model import Binary_model
import matplotlib.pyplot as plt


itterations = 50
accuracies = []

for i in range(itterations):
    model_container = Binary_model()
    model_container.train(100)
    acc = evaluate_model(model_container.model, model_container.test_loader)
    accuracies.append(acc)

def plot_box(data, title="Accuracy of 50 Independently Trained Models", ylabel="accuracy"):
    plt.figure(figsize=(6, 5))
    plt.boxplot(data, vert=True, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    plt.axhline((sum(data)/len(data)), color='red', linestyle='--', label='Mean Accuracy')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_box(accuracies)
