# 🏃‍♂️ Group Activity Recognition (Collective Activity Dataset)

## 📄 Project Overview

This project implements a **Hierarchical CNN + LSTM pipeline** for recognizing **group activities** from video sequences, following the principles described in:  
> "A Hierarchical Deep Temporal Model for Group Activity Recognition"

### **Dataset:**
**Collective Activity Dataset (University of Michigan / Stanford CVGL)**  
Activities: **Crossing, Waiting, Queueing, Walking, Talking**

---

## 📂 Project Structure

datasets/
├── collective_activity_dataset/
│ ├── person_sequences/
│ ├── seq01/
│ ├── seq02/
│ └── ...
models/
├── person_model.py
├── group_model.py
outputs/
├── saved_models/
│ ├── cnn_extractor.pth
│ └── group_model.pth
scripts/
├── train_person_model.py
├── train_group_model.py
├── evaluate_group_model.py
utils/
requirements.txt
README.md

---

## 🚀 Pipeline Architecture (CNN + LSTM)

<p align="center">
  <img src="outputs/group_activity_pipeline.png" alt="Pipeline Diagram" width="600"/>
</p>

---

## 🔧 Installation
```bash
pip install -r requirements.txt
```
---

##  🛠️ Training
1️⃣ Person-Level Action Recognition
```bash

python scripts/train_person_model.py
```
2️⃣ Group-Level Activity Recognition
```bash
python scripts/train_group_model.py
```
---

##📊 Evaluation
```bash
python scripts/evaluate_group_model.py
```
