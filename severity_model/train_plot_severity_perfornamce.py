from dataset.data_fetch import UCIDataset
import matplotlib.pyplot as plt
from severity_model.severity_mlp import evaluate_model
from severity_model.usable_severity_model import Severity_model


def print_predictions(mae_error,all_labels,all_expected):
        plt.figure(figsize=(8, 5))
        plt.scatter([1,2,3,4], [1,2,3,4], alpha=0.7, color="red")
        plt.scatter(all_labels, all_expected, alpha=0.2)
        plt.title("Predicted vs Actual Severity")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


data = UCIDataset()
data.severity_setup()


itterations = 1
all_errors = []

for i in range(itterations):
    model_container = Severity_model()
    model_container.train(num_epochs = 80)
    mae_error, all_expected, all_labels = evaluate_model(model_container.model, model_container.test_loader)
    print_predictions(mae_error, all_labels, all_expected,)
