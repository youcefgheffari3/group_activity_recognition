import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os

from models.person_model import PersonCNNLSTM


# ✅ Your real labels_dict (paste it here)
labels_dict = {
    'seq01': 1, 'seq02': 1, 'seq03': 1, 'seq04': 2, 'seq05': 1, 'seq06': 0,
    'seq07': 1, 'seq08': 0, 'seq09': 2, 'seq10': 0, 'seq11': 3, 'seq12': 2,
    'seq13': 3, 'seq14': 2, 'seq15': 1, 'seq16': 1, 'seq17': 1, 'seq18': 2,
    'seq19': 3, 'seq20': 1, 'seq21': 4, 'seq22': 0, 'seq23': 4, 'seq24': 3,
    'seq25': 3, 'seq26': 3, 'seq27': 4, 'seq28': 1, 'seq29': 3, 'seq30': 1,
    'seq31': 1, 'seq32': 3, 'seq33': 3, 'seq34': 3, 'seq35': 1, 'seq36': 1,
    'seq37': 1, 'seq38': 4, 'seq39': 1, 'seq40': 3, 'seq41': 4, 'seq42': 1,
    'seq43': 1, 'seq44': 2
}


# ✅ Dataset class
class PersonSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, labels_dict, sequence_length=9, transform=None):
        self.root_dir = root_dir
        self.labels_dict = labels_dict
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.person_dirs = sorted(labels_dict.keys())

    def __len__(self):
        return len(self.person_dirs)

    def __getitem__(self, idx):
        person_folder = os.path.join(self.root_dir, self.person_dirs[idx])
        frames = sorted([
            f for f in os.listdir(person_folder)
            if f.endswith('.jpg')
        ])[:self.sequence_length]

        images = []
        for img_name in frames:
            img_path = os.path.join(person_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            images.append(img)

        # If less than 9 images, pad with copies of the last one
        while len(images) < self.sequence_length:
            images.append(images[-1])

        images = torch.stack(images, dim=0)
        label = self.labels_dict[self.person_dirs[idx]]
        return images, torch.tensor(label, dtype=torch.long)


# ✅ Paths and DataLoader
root_dir = 'C:/Users/Gheffari Youcef/Videos/group_activity_recognition/datasets/collective_activity_dataset/person_sequences/'

dataset = PersonSequenceDataset(root_dir=root_dir, labels_dict=labels_dict)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ✅ Model, Loss, Optimizer
model = PersonCNNLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

# ✅ Evaluation Example
model.eval()
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        print(f"True: {labels.tolist()} | Predicted: {predicted.tolist()}")
