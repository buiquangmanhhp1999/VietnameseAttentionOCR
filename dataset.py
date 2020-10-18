import numpy as np
import cv2
import tqdm

from PIL import Image, ImageDraw, ImageFont

import config as cfg

max_len = cfg.seq_len + 1
base_dir = cfg.base_dir
font_path = cfg.font_path


def visualization(image_path, points, label, vis_color=(255, 255, 255)):
    """
    Visualize groundtruth label to image.
    """
    points = np.asarray(points, dtype=np.int32)
    points = np.reshape(points, [-1, 2])
    image = cv2.imread(image_path)
    cv2.polylines(image, [points], 1, (0, 255, 0), 2)
    image = Image.fromarray(image)
    FONT = ImageFont.truetype(font_path, 20, encoding='utf-8')
    DRAW = ImageDraw.Draw(image)

    DRAW.text(points[0], label, vis_color, font=FONT)
    return np.array(image)


def strQ2B(uchar):
    """
    Convert full-width character to half-width character.
    """
    inside_code = ord(uchar)
    if inside_code == 12288:
        inside_code = 32
    elif 65281 <= inside_code <= 65374:
        inside_code -= 65248
    return chr(inside_code)


def preprocess(string):
    """
    Groundtruth label preprocess function.
    """
    # string = [strQ2B(ch) for ch in string.strip()]
    # return ''.join(string)
    return string


class Dataset(object):
    """
    Base class for text dataset preprocess.
    """

    def __init__(self, max_len=max_len, base_dir=base_dir,
                 label_dict=cfg.reverse_label_dict):  # label_dict  label_dict_with_rects 5434+1
        self.label_dict = label_dict
        self.max_len = max_len
        self.base_dir = base_dir
        self.images = []
        self.filenames = []
        self.labels = []
        self.masks = []
        self.bboxes = []
        self.points = []


class OcrDataset(Dataset):
    """
    Custom Dataset for Text Recognition
    """

    def __init__(self, annotation_file='./Data/train_annotation.txt'):
        super(OcrDataset, self).__init__()
        self.label_path = annotation_file

    def load_data(self):
        with open(self.label_path, 'r') as file:
            all_dataset = file.readlines()
        np.random.shuffle(all_dataset)

        for line in tqdm.tqdm(all_dataset):
            img_path, label = line.strip().split('\t', 1)

            if len(label) > self.max_len - 1:
                continue

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]

            skip = False
            for char in label:
                if char not in self.label_dict.keys():
                    skip = True

            if skip:
                continue

            seq_label = []
            for char in label:
                seq_label.append(self.label_dict[char])
            seq_label.append(self.label_dict['EOS'])

            non_zero_count = len(seq_label)
            seq_label = seq_label + [self.label_dict['EOS']] * (self.max_len - non_zero_count)
            mask = [1] * non_zero_count + [0] * (self.max_len - non_zero_count)
            bbox = [[0, 0, h - 1, w - 1]]  # (ymin, xmin, ymax, xmax)
            polygon = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int64)

            self.images.append(img)
            self.filenames.append(img_path)
            self.labels.append(seq_label)
            self.masks.append(mask)
            self.bboxes.append(bbox)
            self.points.append(polygon)


if __name__ == '__main__':
    vn = OcrDataset(annotation_file='./Data/train_annotation.txt')
    vn.load_data()

    filenames = vn.filenames
    labels = vn.labels
    masks = vn.masks
    bboxes = vn.bboxes
    points = vn.points
    images = vn.images

    from sklearn.utils import shuffle

    images, filenames, labels, masks, bboxes, points = shuffle(images, filenames, labels, masks, bboxes, points, random_state=0)

    dataset = {"images": images, "filenames": filenames, "labels": labels, "masks": masks, "bboxes": bboxes, "points": points}
    np.save(cfg.dataset_name, dataset)

