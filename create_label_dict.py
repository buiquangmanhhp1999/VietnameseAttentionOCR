import shutil
import numpy as np

with open('./label_dict/vocab.txt', 'r') as file:
    character = list(file.readline())

with open('./label_dict/vietnamese_label.txt', 'w') as file:
    file.write('{} {}\n'.format(0, 'UNK'))
    file.write('{} {}\n'.format(1, 'EOS'))
    for i, character in enumerate(character, start=2):
        file.write('{} {}\n'.format(i, character))