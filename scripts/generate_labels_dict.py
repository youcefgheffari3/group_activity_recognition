import os
from collections import Counter

root_dir = 'C:/Users/Gheffari Youcef/Videos/group_activity_recognition/datasets/collective_activity_dataset/person_sequences'
sequences = sorted(os.listdir(root_dir))

labels_dict = {}

# Mapping to convert original labels to 0-4 for training
label_mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}

for seq in sequences:
    annotation_file = os.path.join(root_dir, seq, 'annotations.txt')
    if not os.path.isfile(annotation_file):
        continue

    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        class_id = int(parts[-1])
        if class_id in label_mapping:
            labels.append(class_id)

    if labels:
        most_common = Counter(labels).most_common(1)[0][0]
        labels_dict[seq] = label_mapping[most_common]

# Print the final labels_dict
print("Generated labels_dict:")
print(labels_dict)
