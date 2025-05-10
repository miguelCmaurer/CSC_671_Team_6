from binary_model.usable_binary_model import Binary_model
from binary_model.binary_mlp import evaluate_model

model_container = Binary_model()
model_container.train(60)

evaluate_model(model_container.model, model_container.test_loader)
