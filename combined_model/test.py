import torch
import random
import os
from dataset.data_fetch import UCIDataset
from binary_model.binary_mlp import MLP as BinaryMLP
from severity_model.severity_mlp import MLP as SeverityMLP
from combined_model.combined_predictor import Binary_severity
from severity_model.usable_severity_model import Severity_model
from binary_model.usable_binary_model import Binary_model
import matplotlib.pyplot as plt

binary_model_path = "binary_model.pt"
severity_model_path = "severity_model.pt"

if os.path.exists(binary_model_path):
    binary_model = BinaryMLP(num_features=13)
    binary_model.load_state_dict(torch.load(binary_model_path))
    binary_model.eval()
else:
    model_container = Binary_model()
    model_container.train(120)
    torch.save(model_container.model.state_dict(), binary_model_path)
    binary_model = model_container.model

if os.path.exists(severity_model_path):
    severity_model = SeverityMLP(num_features=13)
    severity_model.load_state_dict(torch.load(severity_model_path))
    severity_model.eval()
else:
    model_container = Severity_model()
    model_container.train(60)
    torch.save(model_container.model.state_dict(), severity_model_path)
    severity_model = model_container.model

data = UCIDataset()


combined_model = Binary_severity(binary_model, severity_model)



true_vals = []
pred_vals = []

for _ in range(60):
    features, label = random.choice(data)
    features = features.unsqueeze(0)
    binary_class, severity = combined_model.predict(features)

    true_vals.append(label.item())
    pred_vals.append(severity)


plt.figure(figsize=(10, 5))
x = range(len(true_vals))
plt.plot(x, true_vals, label='True', color="green")
plt.plot(x, pred_vals , label='Predicted', color="tan")
for i in x:
    plt.scatter(i,pred_vals[i],color="tan")
    plt.scatter(i,true_vals[i],color="green")
plt.title("True vs Predicted Severity")
plt.xlabel("Sample Number")
plt.ylabel("Severity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
