#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from model.tensorpack_model_250 import AttentionOCR as model_250
from model.tensorpack_model_330 import AttentionOCR as model_330

from tensorpack.predict import PredictConfig
from tensorpack.tfutils import SmartInit
from tensorpack.tfutils.export import ModelExporter


def export(args):
    model = model_250()
    predcfg = PredictConfig(
        model=model,
        session_init=SmartInit(args.checkpoint_path),
        input_names=model.get_inferene_tensor_names()[0],
        output_names=model.get_inferene_tensor_names()[1])

    ModelExporter(predcfg).export_compact(args.pb_path, optimize=False)
    print(model.get_inferene_tensor_names()[0])
    # ModelExporter(predcfg).export_serving(args.pb_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR')

    parser.add_argument('--pb_path', type=str, help='path to save tensorflow pb model', default='./ckpt_256/model-15000.pb')
    parser.add_argument('--checkpoint_path', type=str, help='path to tensorflow model',
                        default='./ckpt_256/model-15000')

    args = parser.parse_args()
    export(args)
