import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.group_model import CNNFeatureExtractor, GroupActivityLSTM

# -------- Force CPU --------
device = torch.device('cpu')

# -------- Ensure output directory exists --------
save_dir = 'C:/Users/Gheffari Youcef/Videos/group_activity_recognition/outputs/saved_models/'
os.makedirs(save_dir, exist_ok=True)


# -------- Dataset Class --------
class GroupSequenceDataset(Dataset):
    def __init__(self, root_dir, annotations_root, labels_dict, sequence_length=9, transform=None):
        self.root_dir = root_dir
        self.annotations_root = annotations_root
        self.labels_dict = labels_dict
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.sequences = sorted(labels_dict.keys())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        images_path = os.path.join(self.root_dir, seq)
        annotation_file = os.path.join(self.annotations_root, seq, 'annotations.txt')

        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        frame_person_images = {}  # frame_id -> list of person crops
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            seq_num, x, y, w, h, frame_id, class_id = map(int, parts)
            if class_id == -1 or class_id == 1:
                continue  # ignore background or NA
            img_name = f'frame{frame_id:04d}.jpg'
            img_path = os.path.join(images_path, img_name)

            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert('RGB')
            person_crop = img.crop((x, y, x + w, y + h))
            person_crop = self.transform(person_crop)

            if frame_id not in frame_person_images:
                frame_person_images[frame_id] = []
            frame_person_images[frame_id].append(person_crop)

        # Ensure fixed sequence length (pad with last valid)
        frames = sorted(frame_person_images.keys())[:self.sequence_length]
        if not frames:
            raise ValueError(f"No valid annotations for sequence {seq}.")

        while len(frames) < self.sequence_length:
            frames.append(frames[-1])

        per_frame_persons = []
        for frame_id in frames:
            persons = frame_person_images.get(frame_id, [torch.zeros(3, 224, 224)])
            persons = torch.stack(persons, dim=0)
            per_frame_persons.append(persons)

        return per_frame_persons, torch.tensor(self.labels_dict[seq], dtype=torch.long)


# -------- Paths and Labels --------
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

root_dir = 'C:/Users/Gheffari Youcef/Videos/group_activity_recognition/datasets/collective_activity_dataset/person_sequences/'
annotations_root = 'C:/Users/Gheffari Youcef/Videos/group_activity_recognition/datasets/collective_activity_dataset/person_sequences/'

dataset = GroupSequenceDataset(root_dir, annotations_root, labels_dict)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


# -------- Models on CPU --------
cnn_extractor = CNNFeatureExtractor().to(device)
group_model = GroupActivityLSTM().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(cnn_extractor.parameters()) + list(group_model.parameters()), lr=1e-4)


# -------- Training Loop --------
num_epochs = 10
for epoch in range(num_epochs):
    cnn_extractor.train()
    group_model.train()
    running_loss = 0.0

    for per_frame_persons_list, labels in dataloader:
        # batch size = 1
        per_frame_persons_list = per_frame_persons_list[0]
        labels = labels.to(device)

        group_features = []

        for persons in per_frame_persons_list:
            persons = persons.to(device)
            features = cnn_extractor(persons)
            pooled = torch.max(features, dim=0)[0]
            group_features.append(pooled)

        group_features = torch.stack(group_features).unsqueeze(0)  # (1, seq_len, 512)

        outputs = group_model(group_features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")


# -------- Save Models --------
torch.save(cnn_extractor.state_dict(), os.path.join(save_dir, 'cnn_extractor.pth'))
torch.save(group_model.state_dict(), os.path.join(save_dir, 'group_model.pth'))
print(f"\nModels saved successfully to '{save_dir}' ✅")


# -------- Evaluation Example --------
cnn_extractor.eval()
group_model.eval()

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

        print(f"True: {labels.tolist()} | Predicted: {predicted.tolist()}")
