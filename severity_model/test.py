from severity_model.usable_severity_model import Severity_model
from severity_model.severity_mlp import evaluate_model

model_container = Severity_model()
model_container.train(60)

evaluate_model(model_container.model, model_container.test_loader)
