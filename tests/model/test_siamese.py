import torch
from torch import nn
from src.model.siamese import SiameseNetwork

# You don't need to redefine the fixtures here if they're in a conftest.py file in the root directory

def test_model_initialization(model):
    assert isinstance(model, SiameseNetwork)
    assert isinstance(model.resnet, nn.Sequential)
    assert isinstance(model.fc, nn.Linear)
    assert isinstance(model.classifier, nn.Linear)

def test_model_forward(model, anchor, positive, negative):
    anchor_out, positive_out, negative_out, anchor_class = model(anchor, positive, negative)

    assert anchor_out.shape == (1, 128)
    assert positive_out.shape == (1, 128)
    assert negative_out.shape == (1, 128)
    assert anchor_class.shape == (1, 10)  # Assuming num_classes=10

def test_model_output_types(model, anchor, positive, negative):
    anchor_out, positive_out, negative_out, anchor_class = model(anchor, positive, negative)

    assert isinstance(anchor_out, torch.Tensor)
    assert isinstance(positive_out, torch.Tensor)
    assert isinstance(negative_out, torch.Tensor)
    assert isinstance(anchor_class, torch.Tensor)

def test_embedding_normalization(model, anchor):
    anchor_out, _, _, _ = model(anchor, anchor, anchor)  # Using anchor for all inputs

    # Check if the embedding is normalized (L2 norm should be close to 1)
    norm = torch.norm(anchor_out, p=2, dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

def test_classification_output(model, anchor, labels):
    _, _, _, anchor_class = model(anchor, anchor, anchor)  # Using anchor for all inputs

    assert anchor_class.shape[1] == 10  # Assuming num_classes=10
    assert torch.argmax(anchor_class, dim=1).item() in range(10)
