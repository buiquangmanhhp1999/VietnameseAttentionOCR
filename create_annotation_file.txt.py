from pathlib import Path
import os


train_data_path = list(Path('./Data/out/cmnd/add/').glob(".jpg")) + list(Path('./Data/out/cmnd/name/').glob("*.jpg"))

with open('./Data/train_annotation.txt', 'w') as file:
    for img_path in train_data_path:
        name = img_path.name[:-4]
        true_label = name.split('_')[0]  # [string_label]_[idx].jpg
        true_label = true_label.strip()
        file.write(str(img_path) + '\t' + true_label + '\n')

