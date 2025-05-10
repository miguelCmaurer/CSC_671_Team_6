import torch

class Binary_severity(torch.nn.Module):
    def __init__(self, binary_mlp, severity_mlp):
        super().__init__()
        self.binary = binary_mlp
        self.severity = severity_mlp

    def forward(self , x):
        binary_pred = self.binary(x)
        severity_pred = self.severity(x)
        return binary_pred, severity_pred

    def predict(self, point):
        self.eval()
        with torch.no_grad():
            binary_out, severity_out = self.forward(point)
            binary_class = binary_out.argmax(dim=1).item()
            if binary_class == 1:
                return binary_class, severity_out.item()
            else:
                return binary_class, 0
