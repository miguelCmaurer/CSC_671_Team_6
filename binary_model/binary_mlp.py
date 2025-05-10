import torch

class MLP(torch.nn.Module):
    def __init__(self, num_features, dropout=0.2):
        super().__init__()
        self.all_layers = torch.nn.Sequential(
            torch.nn.Linear(num_features, 30),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(30, 80),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(80, 2)
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
        return  prediction.argmax(dim=1).item()

def train_model(model, dataloader, loss_fn, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels in dataloader:
            labels = labels.view(-1).long()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f}")


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels.view(-1).long()).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy
