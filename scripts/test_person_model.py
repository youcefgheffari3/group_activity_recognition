import torch
from torchvision import transforms
from PIL import Image
import os
from models.person_model import PersonCNNLSTM

# Dummy input: 9 frames of 3x224x224 images
dummy_input = torch.randn(1, 9, 3, 224, 224)  # (batch, sequence, channels, height, width)

model = PersonCNNLSTM()
output = model(dummy_input)
print("Output shape:", output.shape)
