import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PersonSequenceDataset(Dataset):
    def __init__(self, root_dir, annotations_root, sequence_length=9, transform=None):
        self.root_dir = root_dir
        self.annotations_root = annotations_root
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.sequences = self._gather_sequences()

    def _gather_sequences(self):
        sequences = []

        for seq_folder in sorted(os.listdir(self.root_dir)):
            seq_path = os.path.join(self.root_dir, seq_folder)
            annotation_file = os.path.join(self.annotations_root, seq_folder, 'annotations.txt')

            if not os.path.exists(annotation_file):
                continue

            with open(annotation_file, 'r') as f:
                lines = f.readlines()

            frame_person_data = {}
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                seq_num, x, y, w, h, frame_id, class_id = map(int, parts)
                if class_id == -1 or class_id == 1:
                    continue  # ignore NA/background

                img_name = f'frame{frame_id:04d}.jpg'
                img_path = os.path.join(seq_path, img_name)

                if not os.path.exists(img_path):
                    continue

                if frame_id not in frame_person_data:
                    frame_person_data[frame_id] = []

                frame_person_data[frame_id].append({
                    'img_path': img_path,
                    'bbox': (x, y, x + w, y + h),
                    'class_id': class_id,
                })

            frames = sorted(frame_person_data.keys())[:self.sequence_length]

            # pad missing frames
            while len(frames) < self.sequence_length:
                frames.append(frames[-1])

            sequences.append({
                'frames': frames,
                'data': frame_person_data,
                'seq_folder': seq_folder
            })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        frames = seq_info['frames']
        frame_person_data = seq_info['data']

        person_images = []
        labels = []

        for frame_id in frames:
            persons = frame_person_data.get(frame_id, [])

            if len(persons) == 0:
                crop = torch.zeros(3, 224, 224)
                person_images.append(crop)
                labels.append(-1)  # no person
                continue

            # take only the first person (simplification)
            person_info = persons[0]
            img = Image.open(person_info['img_path']).convert('RGB')
            crop = img.crop(person_info['bbox'])
            crop = self.transform(crop)
            person_images.append(crop)
            labels.append(person_info['class_id'])

        images_tensor = torch.stack(person_images, dim=0)  # (sequence_length, 3, 224, 224)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return images_tensor, labels_tensor
