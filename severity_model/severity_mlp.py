import torch
from sklearn.metrics import mean_absolute_error

class MLP(torch.nn.Module):
    def __init__(self, num_features, dropout=0.2):
        super().__init__()
        self.all_layers = torch.nn.Sequential(
            torch.nn.Linear(num_features, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1)
        )

    def forward(self , x):
        logits = self.all_layers(x)
        return logits

    def predict(self, point, dataset):
        self.eval()
        point = (point - dataset.x_mean) / dataset.x_std
        with torch.no_grad():
            prediction = self.forward(point)
        print(prediction)
        return  prediction.item()

def train_model(model, dataloader, loss_fn, optimizer, num_epochs=75):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels in dataloader:
            labels = labels.view(-1,1).float()
            outputs = model(features)
            loss = loss_fn(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}/{num_epochs} - loss: {total_loss:.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    all_expected = []
    all_labels = []
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features).view(-1)
            all_expected += outputs.tolist()
            all_labels += labels.view(-1).tolist()
    mae = mean_absolute_error(all_labels, all_expected)
    print(f"Mean Absolute Error: {mae:.3f}")
    return mae, all_expected, all_labels
