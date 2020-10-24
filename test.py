import cv2
import argparse
import config_330
import config_250
from pathlib import Path
import tensorflow as tf
import numpy as np
import time
import os


class TextRecognition(object):
    """
    AttentionOCR with tensorflow pb model.
    """

    def __init__(self, pb_file_330, pb_file_230, seq_len):
        self.pb_file_330 = pb_file_330
        self.pb_file_230 = pb_file_230
        self.seq_len = seq_len

        # load model
        self.graph_230, self.sess_230, self.img_ph_230, self.label_ph_230, self.is_training_230, self.dropout_230, self.preds_230, self.probs_230 = self.load_model(model_width=230)
        self.graph_330, self.sess_330, self.img_ph_330, self.label_ph_330, self.is_training_330, self.dropout_330, self.preds_330, self.probs_330 = self.load_model(model_width=330)

    def load_model(self, model_width):
        if model_width == 230:
            pb_file = self.pb_file_230
        elif model_width == 330:
            pb_file = self.pb_file_330
        else:
            raise ValueError("Model width invalid")

        # Load the protobuf file from disk and parse it to retrieve the unserialized graph_def
        with tf.gfile.GFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # then, we import graph_def into a new graph and return it
        with tf.Graph().as_default() as new_graph:
            tf.import_graph_def(graph_def, name='')

        sess = tf.Session(graph=new_graph)
        img_ph = sess.graph.get_tensor_by_name('image:0')
        label_ph = sess.graph.get_tensor_by_name('label:0')
        is_training = sess.graph.get_tensor_by_name('is_training:0')
        dropout = sess.graph.get_tensor_by_name('dropout_keep_prob:0')
        preds = sess.graph.get_tensor_by_name('sequence_preds:0')
        probs = sess.graph.get_tensor_by_name('sequence_probs:0')

        return new_graph, sess, img_ph, label_ph, is_training, dropout, preds, probs

    @staticmethod
    def preprocess(image, model_width):
        """
        Preprocess for test.
        Args:
            model_width:
            image: test image
        """
        height, width = image.shape[:2]

        padding_top = 0
        padding_down = 0
        padding_left = 0
        padding_right = model_width - width

        if padding_right > 0:
            image = cv2.copyMakeBorder(image, padding_top, padding_down, padding_left, padding_right,
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image = cv2.resize(image, (model_width, config_330.height_img))
        image = image / 255.
        return image

    def batch_preprocess(self, batch_images):
        thresh = 30

        images_width = np.array([img.shape[1] for img in batch_images])
        mask = images_width < (config_250.image_size + thresh)
        batch_230_images = []
        batch_330_images = []

        for i, status in enumerate(mask):
            if status:
                batch_230_images.append(batch_images[i])
                # print("Shape 230: ", np.array(batch_images[i]).shape)
            else:
                batch_330_images.append(batch_images[i])
                # print("Shape 330: ", np.array(batch_images[i]).shape)

        batch_230_images = np.array([self.preprocess(img, model_width=config_250.image_size) for img in batch_230_images])
        batch_330_images = np.array([self.preprocess(img, model_width=config_330.image_size) for img in batch_330_images])
        return batch_230_images, batch_330_images

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

        probabilities = probs[:min(len(results) + 1, config_330.seq_len + 1)]
        return ''.join(results), np.mean(probabilities)

    def predict(self, images, model_width):
        before = time.time()
        if model_width == 230:
            pred_sentences, pred_probs = self.sess_230.run([self.preds_230, self.probs_230],
                                                           feed_dict={self.is_training_230: False, self.dropout_230: 1.0,
                                                                      self.img_ph_230: images,
                                                                      self.label_ph_230: np.ones((1, self.seq_len),
                                                                                             np.int32)})
        else:
            pred_sentences, pred_probs = self.sess_330.run([self.preds_330, self.probs_330],
                                                           feed_dict={self.is_training_330: False, self.dropout_330: 1.0,
                                                                      self.img_ph_330: images,
                                                                      self.label_ph_330: np.ones((1, self.seq_len),
                                                                                             np.int32)})
        after = time.time()
        for pred_sentence, pred_prob in zip(pred_sentences, pred_probs):
            text, confidence = self.label2str(pred_sentence, pred_prob, config_330.label_dict)
            print("-------------------------------------------------------------")
            print("text: ", text)
            print("confidence: ", confidence)
        return after - before

    def predict_on_batch(self, batch_images):
        batch_230_images, batch_330_images = self.batch_preprocess(batch_images)
        ti_1 = ti_2 = 0
        if np.array(batch_230_images).shape[0] > 0:
            ti_1 = self.predict(batch_230_images, model_width=230)

        if np.array(batch_330_images).shape[0] > 0:
            ti_2 = self.predict(batch_330_images, model_width=330)

        print("Total time: ", ti_1 + ti_2)

    def test_pb_file(self, args, model_width):
        if model_width == 256:
            sess = self.sess_230
            preds = self.preds_230
            probs = self.probs_230
            is_training = self.is_training_230
            img_ph = self.img_ph_230
            label_ph = self.label_ph_230
            dropout = self.dropout_230
        else:
            sess = self.sess_330
            preds = self.preds_330
            probs = self.probs_330
            is_training = self.is_training_330
            img_ph = self.img_ph_330
            label_ph = self.label_ph_330
            dropout = self.dropout_330

        result_file_name = Path(args.pb_path_256).name[:-3]
        with open("result/" + result_file_name + '.txt', "w") as f:
            ned = 0.
            count = 0
            for filename in os.listdir(args.img_folder):
                # read image
                img_path = os.path.join(args.img_folder, filename)
                print("----> image path: ", img_path)
                name = filename.split('_')[0]
                image = cv2.imread(img_path)

                # preprocess image
                image = self.preprocess(image, model_width)
                image = np.expand_dims(image, 0)
                before = time.time()
                pred_sentences, pred_probs = sess.run([preds, probs],
                                                      feed_dict={is_training: False, dropout: 1.0,
                                                                 img_ph: image,
                                                                 label_ph: np.ones((1, self.seq_len), np.int32)})
                after = time.time()
                required_time = after - before
                text, confidence = self.label2str(pred_sentences[0], pred_probs[0], config_330.label_dict)

                cal_slim = self.cal_sim(text, name)
                ned += cal_slim
                count += 1

                print("Text: ", text)
                print("Label: ", name)
                print("confidence: ", confidence)
                print("cal_sim: ", cal_slim)
                print("Time: ", required_time)
                print("-------------------------------")

                # write to file
                self.write_to_file(f, img_path, text, name, confidence, required_time, cal_slim)
            f.write("Total {} Images | Average NED: {}".format(count, ned / count))

    @staticmethod
    def write_to_file(f, path, text, name, confidence, required_time, cal_slim):
        f.write("Path: {}".format(path))
        f.write("\n")
        f.write("Text: {}".format(text))
        f.write("\n")
        f.write("Label: {}".format(name))
        f.write("\n")
        f.write("Confidence: {}".format(confidence))
        f.write("\n")
        f.write("Time: {}".format(required_time))
        f.write("\n")
        f.write("1-N.E.D: {}".format(cal_slim))
        f.write("\n")
        f.write("---------------------------------------------")
        f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR')
    parser.add_argument('--pb_path_330', type=str, help='path to tensorflow pb model',
                        default='./ckpt_330/model-40000.pb')
    parser.add_argument('--pb_path_256', type=str, help='path to tensorflow pb model',
                        default='./ckpt_256/model-15000.pb')
    parser.add_argument('--checkpoint_path', type=str, help='path to tensorflow pb model',
                        default='./ckpt/model-40000')
    parser.add_argument('--img_folder', type=str, help='path to image folder', default='./test/')

    args = parser.parse_args()

    model = TextRecognition(pb_file_330=args.pb_path_330, pb_file_230=args.pb_path_256, seq_len=config_330.seq_len + 1)
    # model.test_pb_file(args)
    img1 = cv2.imread('./test/An Hòa_4.jpg')
    img2 = cv2.imread('./test/Phường An Hòa_5.jpg')
    img3 = cv2.imread('./test/Phường Cái Khế_1.jpg')
    img4 = cv2.imread('./test/Phường Thới Bình_9.jpg')
    img5 = cv2.imread('./test/Thới Bình_8.jpg')
    img6 = cv2.imread('./test/Cái Khế, Ninh Kiều, Cần Thơ_2.jpg')
    batch_imgs = [img1, img2, img3, img4, img5, img6]
    # model.predict_on_batch(batch_imgs)
    # model.predict_on_batch(batch_imgs)
    # model.predict_on_batch(batch_imgs)
    model.test_pb_file(args, model_width=256)
