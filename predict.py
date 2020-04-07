import os
import cv2
import numpy as np
import tensorflow as tf
import config
import data_setup
import model_fn


class PredictModes:
    def __init__(self, args):
        self.args = args
        self.model_fn_params = {'batch_norm': self.args.no_bn, 'dropout': 0., 'classes': self.args.classes}
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True  # Allow full memory usage of GPU.
        self.load_path = config.paths['save'] + '/' + self.args.load_model
        self.ynet = tf.estimator.Estimator(model_fn=model_fn.unet_model_fn,
                                           model_dir=self.load_path,
                                           params=self.model_fn_params)

    def run_chaos_test(self):
        pred_input_fn = data_setup.DatasetHandler('chaos-test', self.args)
        predicted = self.ynet.predict(input_fn=lambda: pred_input_fn.input_fn(),
                                      predict_keys=['output_img', 'path'],
                                      yield_single_examples=True)
        for idx, output in enumerate(predicted):
            path = output['path'].decode("utf-8")
            if self.args.modality == 'ALL':
                new_path = path.replace('Test_Sets', 'Task1')
            elif self.args.modality == 'CT':
                new_path = path.replace('Test_Sets', 'Task2')
            else:
                new_path = path.replace('Test_Sets', 'Task3')
            if 'CT' in new_path:
                intensity = 255
            else:  # 'MR' in new_path:
                intensity = 63
            results = output['output_img'].astype(np.uint8) * intensity
            new_path = new_path.replace('DICOM_anon', 'Results')
            new_path = new_path.replace('.dcm', '.png')
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            cv2.imwrite(new_path, results)

    def predict(self):
        pred_input_fn = data_setup.DatasetHandler('predict', self.args)
        predicted = self.ynet.predict(input_fn=lambda: pred_input_fn.input_fn(),
                                      predict_keys=['output_img', 'path'],
                                      yield_single_examples=True)
        for idx, output in enumerate(predicted):
            path = output['path'].decode("utf-8")
            new_path = path.replace('DICOM_anon', 'Results')
            new_path = new_path.replace('.dcm', '.png')
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            results = output['output_img'].astype(np.uint8) * 255
            cv2.imwrite(new_path, results)
