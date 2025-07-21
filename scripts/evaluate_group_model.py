import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from PIL import Image
from models.group_model import CNNFeatureExtractor, GroupActivityLSTM
from train_group_model import GroupSequenceDataset, labels_dict


# -------- Force CPU --------
device = torch.device('cpu')


# -------- Paths --------
root_dir = 'C:/Users/Gheffari Youcef/Videos/group_activity_recognition/datasets/collective_activity_dataset/person_sequences/'
annotations_root = 'C:/Users/Gheffari Youcef/Videos/group_activity_recognition/datasets/collective_activity_dataset/person_sequences/'
save_dir = 'C:/Users/Gheffari Youcef/Videos/group_activity_recognition/outputs/saved_models/'

# -------- Dataset --------
dataset = GroupSequenceDataset(root_dir, annotations_root, labels_dict)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


# -------- Load Models --------
cnn_extractor = CNNFeatureExtractor().to(device)
group_model = GroupActivityLSTM().to(device)

cnn_extractor.load_state_dict(torch.load(os.path.join(save_dir, 'cnn_extractor.pth'), map_location=device))
group_model.load_state_dict(torch.load(os.path.join(save_dir, 'group_model.pth'), map_location=device))

cnn_extractor.eval()
group_model.eval()

# -------- Evaluation Loop --------
all_preds = []
all_labels = []

with torch.no_grad():
    for per_frame_persons_list, labels in dataloader:
        per_frame_persons_list = per_frame_persons_list[0]
        labels = labels.to(device)

        group_features = []
        for persons in per_frame_persons_list:
            persons = persons.to(device)
            features = cnn_extractor(persons)
            pooled = torch.max(features, dim=0)[0]
            group_features.append(pooled)

        group_features = torch.stack(group_features).unsqueeze(0)

        outputs = group_model(group_features)
        predicted = torch.argmax(outputs, dim=1)

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())


# -------- Compute Accuracy & Confusion Matrix --------
accuracy = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n✅ Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", cm)
