#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import argparse
from model.tensorpack_model import *
import config as cfg
import tensorflow as tf
from common import polygons_to_mask
from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit


class TextRecognition(object):
    """
    AttentionOCR with tensorflow pb model.
    """

    def __init__(self, pb_file, seq_len):
        self.pb_file = pb_file
        self.seq_len = seq_len

        # load model
        self.graph = self.load_model()

    def load_model(self):
        # Load the protobuf file from disk and parse it to retrieve the unserialized graph_def
        with tf.gfile.GFile(self.pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # then, we import graph_def into a new graph and return it
        with tf.Graph().as_default() as new_graph:
            tf.import_graph_def(graph_def, name='')

        return new_graph

    @staticmethod
    def preprocess(image, points, size=cfg.image_size):
        """
        Preprocess for test.
        Args:
            image: test image
            points: text polygon
            size: test image size
        """
        height, width = image.shape[:2]
        mask = polygons_to_mask([np.asarray(points, np.float32)], height, width)
        x, y, w, h = cv2.boundingRect(mask)
        mask = np.expand_dims(np.float32(mask), axis=-1)
        image = image * mask
        image = image[y:y + h, x:x + w, :]

        new_height, new_width = (size, int(w * size / h)) if h > w else (int(h * size / w), size)
        image = cv2.resize(image, (new_width, new_height))

        if new_height > new_width:
            padding_top, padding_down = 0, 0
            padding_left = (size - new_width) // 2
            padding_right = size - padding_left - new_width
        else:
            padding_left, padding_right = 0, 0
            padding_top = (size - new_height) // 2
            padding_down = size - padding_top - new_height

        image = cv2.copyMakeBorder(image, padding_top, padding_down, padding_left, padding_right,
                                   borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        image = image / 255.
        return image

    @staticmethod
    def cal_sim(str1, str2):
        """
            Normalized Edit Distance metric (1-N.E.D specifically)
        """
        m = len(str1) + 1
        n = len(str2) + 1
        matrix = np.zeros((m, n))
        for i in range(m):
            matrix[i][0] = i

        for j in range(n):
            matrix[0][j] = j

        for i in range(1, m):
            for j in range(1, n):
                if str1[i - 1] == str2[j - 1]:
                    matrix[i][j] = matrix[i - 1][j - 1]
                else:
                    matrix[i][j] = min(matrix[i - 1][j - 1], min(matrix[i][j - 1], matrix[i - 1][j])) + 1

        lev = matrix[m - 1][n - 1]
        if (max(m - 1, n - 1)) == 0:
            sim = 1.0
        else:
            sim = 1.0 - lev / (max(m - 1, n - 1))
        return sim

    @staticmethod
    def label2str(preds, probs, label_dict, eos='EOS'):
        """
        Predicted sequence to string.
        """
        results = []
        for idx in preds:
            if label_dict[idx] == eos:
                break
            results.append(label_dict[idx])

        probabilities = probs[:min(len(results) + 1, cfg.seq_len + 1)]
        return ''.join(results), np.mean(probabilities)

    def test_pb_file(self, args):
        with tf.Session(graph=self.graph) as sess:
            img_ph = sess.graph.get_tensor_by_name('image:0')
            label_ph = sess.graph.get_tensor_by_name('label:0')
            is_training = sess.graph.get_tensor_by_name('is_training:0')
            dropout = sess.graph.get_tensor_by_name('dropout_keep_prob:0')
            preds = sess.graph.get_tensor_by_name('sequence_preds:0')
            probs = sess.graph.get_tensor_by_name('sequence_probs:0')

            with open("result/model-30000.txt", "w") as f:
                ned = 0.
                count = 0
                for filename in os.listdir(args.img_folder):
                    # read image
                    img_path = os.path.join(args.img_folder, filename)
                    print("----> image path: ", img_path)
                    name = filename.split('_')[0]
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    height, width = image.shape[:2]
                    points = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]

                    # preprocess image
                    image = self.preprocess(image, points, cfg.image_size)
                    image = np.expand_dims(image, 0)
                    before = time.time()
                    pred_sentences, pred_probs = sess.run([preds, probs],
                                                          feed_dict={is_training: False, dropout: 1.0,
                                                                     img_ph: image,
                                                                     label_ph: np.ones((1, self.seq_len),
                                                                                            np.int32)})
                    after = time.time()
                    required_time = after - before
                    text, confidence = self.label2str(pred_sentences[0], pred_probs[0], cfg.label_dict)

                    cal_slim = self.cal_sim(text, name)
                    ned += cal_slim
                    count += 1

                    print("Text: ", text)
                    print("Label: ", name)
                    print("confidence: ", confidence)
                    print("cal_sim: ", cal_slim)
                    print("Time: ", after - before)
                    print("-------------------------------")

                    # write to file
                    self.write_to_file(f, img_path, text, name, confidence, required_time, cal_slim)
                f.write("Total {} Images | Average NED: {}".format(count, ned / count))

    def test_checkpoint(self, args):
        model = AttentionOCR()
        predcfg = PredictConfig(
            model=model,
            session_init=SmartInit(args.checkpoint_path),
            input_names=model.get_inferene_tensor_names()[0],
            output_names=model.get_inferene_tensor_names()[1])

        predictor = OfflinePredictor(predcfg)
        with open("result/model-30000.txt", "w") as f:
            ned = 0.
            count = 0
            for filename in os.listdir(args.img_folder):
                img_path = os.path.join(args.img_folder, filename)
                print("----> image path: ", img_path)
                name = filename.split('_')[0]
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                height, width = image.shape[:2]
                points = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]

                image = self.preprocess(image, points, cfg.image_size)
                before = time.time()
                preds, probs = predictor(np.expand_dims(image, axis=0), np.ones([1, cfg.seq_len + 1], np.int32), False,
                                         1.)

                after = time.time()
                text, confidence = self.label2str(preds[0], probs[0], cfg.label_dict)
                cal_slim = self.cal_sim(text, name)

                ned += cal_slim
                count += 1
                required_time = after - before

                print("Text: ", text)
                print("Label: ", name)
                print("confidence: ", confidence)
                print("cal_sim: ", cal_slim)
                print("Time: ", time)
                print("-------------------------------")

                # write to file
                self.write_to_file(f, img_path, text, name, confidence, required_time, cal_slim)
            f.write("Total {} Images | Average NED: {}".format(count, ned / count))

    @staticmethod
    def write_to_file(f, path, text, name, confidence, time, cal_slim):
        f.write("Path: {}".format(path))
        f.write("\n")
        f.write("Text: {}".format(text))
        f.write("\n")
        f.write("Label: {}".format(name))
        f.write("\n")
        f.write("Confidence: {}".format(confidence))
        f.write("\n")
        f.write("Time: {}".format(time))
        f.write("\n")
        f.write("1-N.E.D: {}".format(cal_slim))
        f.write("\n")
        f.write("---------------------------------------------")
        f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR')

    parser.add_argument('--pb_path', type=str, help='path to tensorflow pb model', default='./ckpt/model-30000.pb')
    parser.add_argument('--checkpoint_path', type=str, help='path to tensorflow pb model',
                        default='./ckpt/model-30000')
    parser.add_argument('--img_folder', type=str, help='path to image folder', default='./test/')

    args = parser.parse_args()

    model = TextRecognition(args.pb_path, cfg.seq_len + 1)
    model.test_pb_file(args)
