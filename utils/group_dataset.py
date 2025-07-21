import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


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
