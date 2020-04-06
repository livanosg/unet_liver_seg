from glob import glob

import numpy as np
from cv2.cv2 import imread
from pydicom import dcmread

from augmentations import Augmentations
from config import paths


class DatasetHandler:
    def __init__(self, dataset, args):
        self.modality = args.modality
        self.dataset = dataset
        self.classes = args.classes
        self.batch_size = args.batch_size
        self.augm_prob = args.augm_prob
        if self.dataset in ('eval', 'pred',):
            self.batch_size = 1
            self.augm_prob = 0.

    def __len__(self):
        return len(self.get_dataset_paths())

    def get_ct_list(self):
        dirs = {'ct_dcm_paths': '/**/CT/**/**.dcm',
                'ct_grd_paths': '/**/CT/**/*.png'}
        for key in dirs.keys():
            dirs[key] = sorted(glob(paths[self.dataset] + dirs[key], recursive=True))
        return dirs['ct_dcm_paths'], dirs['ct_grd_paths']

    def get_mr_list(self):
        dirs = {'mr_dcm_in': '/**/MR/**/InPhase/*.dcm', 'mr_dcm_out': '/**/MR/**/OutPhase/*.dcm',
                'mr_grd_t1': '/**/MR/**/T1DUAL/**/*.png', 'mr_dcm_t2': '/**/MR/**/T2SPIR/**/*.dcm',
                'mr_grd_t2': '/**/MR/**/T2SPIR/**/*.png'}
        # noinspection PyUnresolvedReferences
        for key in dirs.keys():
            dirs[key] = sorted(glob(paths[self.dataset] + dirs[key], recursive=True))
        if self.dataset == 'chaos-test':
            mr_dicom_list = dirs['mr_dcm_in'] + dirs['mr_dcm_t2']  # todo in/out are registered train accordingly
        else:
            mr_dicom_list = dirs['mr_dcm_in'] + dirs['mr_dcm_out'] + dirs['mr_dcm_t2']
        mr_ground_list = dirs['mr_grd_t1'] + dirs['mr_grd_t1'] + dirs['mr_grd_t2']
        return mr_dicom_list, mr_ground_list

    def get_dataset_paths(self):
        ct_dicom_list, ct_ground_list = self.get_ct_list()
        mr_dicom_list, mr_ground_list = self.get_mr_list()

        if self.dataset in ('train', 'eval'):
            assert len(ct_dicom_list) == len(ct_ground_list)  # Check lists length
            ct_data_path_list = list(zip(ct_dicom_list, ct_ground_list))
            assert len(mr_dicom_list) == len(mr_ground_list)  # Check lists length
            mr_data_path_list = list(zip(mr_dicom_list, mr_ground_list))
        else:
            ct_data_path_list = ct_dicom_list
            mr_data_path_list = mr_dicom_list

        if self.modality == 'CT':
            return ct_data_path_list
        elif self.modality == 'MR':
            return mr_data_path_list
        elif self.modality == 'ALL':
            return ct_data_path_list + mr_data_path_list
        else:
            raise ValueError()

    def dataset_generator(self):
        data_paths = self.get_dataset_paths()
        if self.dataset in ('train', 'eval'):
            np.random.shuffle(data_paths)
            for dicom_path, label_path in data_paths:
                dicom, label = dcmread(dicom_path).pixel_array, imread(label_path, 0)
                if 'MR' in label_path:
                    label[label != 63] = 0
                if self.dataset == 'train':  # Data augmentation
                    resize = {'MR': 320 - dicom.shape[0], 'ALL': 512 - dicom.shape[0], 'CT': 512 - dicom.shape[0]}
                    dicom = np.pad(dicom, [int(resize[self.modality] / 2)], mode='constant',
                                   constant_values=np.min(dicom))
                    label = np.pad(label, [int(resize[self.modality] / 2)], mode='constant',
                                   constant_values=np.min(label))
                    if np.random.random() < self.augm_prob:
                        augmentation = Augmentations()
                        dicom, label = augmentation(input_image=dicom, label=label)
                dicom = (dicom - np.mean(dicom)) / np.std(dicom)  # Normalize
                label[label > 0] = 1
                yield dicom, label
        else:
            for dicom_path in data_paths:
                dicom = dcmread(dicom_path).pixel_array
                yield (dicom - np.mean(dicom)) / np.std(dicom), dicom_path

    def input_fn(self):
        import tensorflow as tf
        if self.dataset in ('train', 'eval'):
            data_set = tf.data.Dataset.from_generator(generator=lambda: self.dataset_generator(),
                                                      output_types=(tf.float32, tf.int32),
                                                      output_shapes=(
                                                          tf.TensorShape([None, None]), tf.TensorShape([None, None])))
            data_set = data_set.map(
                lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=self.classes, dtype=tf.float32)))
            data_set = data_set.map(lambda x, y: (tf.expand_dims(tf.cast(x, tf.float32), -1), y))
            data_set = data_set.map(lambda x, y: ({'image': x}, {'label': y}))
            if self.dataset == 'train':
                data_set = data_set.batch(self.batch_size)
                data_set = data_set.repeat()
            if self.dataset == 'eval':
                data_set = data_set.batch(1)
        else:
            data_set = tf.data.Dataset.from_generator(generator=lambda: self.dataset_generator(),
                                                      output_types=(tf.float32, tf.string),
                                                      output_shapes=(
                                                          tf.TensorShape([None, None]), tf.TensorShape(None)))
            data_set = data_set.map(lambda x, y: {'image': tf.expand_dims(x, -1), 'path': tf.cast(y, tf.string)})
            data_set = data_set.batch(1)
            data_set = data_set.repeat(1)

        data_set = data_set.prefetch(buffer_size=-1)
        return data_set
