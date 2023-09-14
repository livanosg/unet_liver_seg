import os
from glob import glob
import numpy as np
from cv2.cv2 import imread, resize, copyMakeBorder,  INTER_NEAREST, BORDER_CONSTANT
from pydicom import dcmread
from augmentations import Augmentations
from config import TRAIN_DIR, EVAL_DIR, TEST_DIR


class DatasetHandler:
    def __init__(self, mode, args):
        self.modality = args.modality
        self.mode = mode
        self.classes = args.classes
        if self.mode == "train":
            self.batch_size = args.batch_size
            self.augm_prob = args.augm_prob
            self.data_dir = TRAIN_DIR
        else:
            self.batch_size = 1
            self.augm_prob = 0.
            if self.mode == "eval":
                self.data_dir = EVAL_DIR
            else:
                self.data_dir = TEST_DIR

    def __len__(self):
        return len(self.get_dataset_paths())

    def get_ct_paths(self):
        ct_dcm = sorted(glob(os.path.join(self.data_dir, "CT", "**/**.dcm"), recursive=True))
        if self.mode != "pred":
            ct_grd = sorted(glob(os.path.join(self.data_dir, "CT", "**/**.png"), recursive=True))
            return list(zip(ct_dcm, ct_grd))
        else:
            return ct_dcm

    def get_mr_paths(self):
        mr_t1_in = sorted(glob(os.path.join(self.data_dir, "MR", "**", "T1DUAL", "**", "InPhase", "**.dcm"), recursive=True))
        mr_t1_out = sorted(glob(os.path.join(self.data_dir, "MR", "**", "T1DUAL", "**", "OutPhase", "**.dcm"), recursive=True))
        mr_t2 = sorted(glob(os.path.join(self.data_dir, "MR", "**", "T2SPIR", "**", "**.dcm"), recursive=True))
        mr_dcm = mr_t1_in + mr_t1_out + mr_t2

        if self.mode != "pred":
            grd_t1 = sorted(glob(os.path.join(self.data_dir, "MR", "**", "T1DUAL", "Ground", "**.png"), recursive=True))
            grd_t2 = sorted(glob(os.path.join(self.data_dir, "MR", "**", "T2SPIR", "**", "**.png"), recursive=True))
            mr_grd = grd_t1 + grd_t1 + grd_t2
            return list(zip(mr_dcm, mr_grd))
        else:
            return mr_dcm

    def get_dataset_paths(self):
        if self.modality == 'CT':
            return self.get_ct_paths()
        elif self.modality == 'MR':
            return self.get_mr_paths()
        elif self.modality == 'ALL':
            return self.get_ct_paths() + self.get_mr_paths()
        else:
            raise ValueError()

    def dataset_generator(self):
        data_paths = self.get_dataset_paths()
        if self.mode in ('train', 'eval'):
            np.random.shuffle(data_paths)
            for dicom_path, label_path in data_paths:
                image, label = dcmread(dicom_path).pixel_array, imread(label_path, 0)
                size_ratio = 224 / max(np.shape(image)[:-1])
                image = resize(image, dsize=None, fx=size_ratio, fy=size_ratio, interpolation=INTER_NEAREST)
                dx = (image.shape[0] - image.shape[1]) / 2  # Compare height-width
                tblr = [int(np.ceil(np.abs(dx))), int(np.floor(np.abs(dx))), 0, 0]  # Pad top-bottom
                if dx > 0:  # If height > width
                    tblr = tblr[2:] + tblr[:2]  # Pad left-right
                image = copyMakeBorder(image, *tblr, borderType=BORDER_CONSTANT)  # Pad with zeros to make it squared.
                label = resize(label, dsize=None, fx=size_ratio, fy=size_ratio, interpolation=INTER_NEAREST)
                label = copyMakeBorder(label, *tblr, borderType=BORDER_CONSTANT)  # Pad with zeros to make it squared.
                image = (image - np.mean(image)) / np.std(image)  # Standardize
                if 'MR' in label_path:  # Zero out irrelevant masks
                    label[label < 55] = 0
                    label[label > 70] = 0
                    label[label > 0] = 1
                if self.mode == 'train':  # Data augmentation
                    if np.random.random() < self.augm_prob:
                        augmentation = Augmentations()
                        image, label = augmentation(input_image=image, label=label)
                yield image, label
        else:
            for dicom_path in data_paths:
                image = dcmread(dicom_path).pixel_array
                yield (image - np.mean(image)) / np.std(image), dicom_path

    def input_fn(self):
        import tensorflow as tf
        if self.mode in ('train', 'eval'):
            data_set = tf.data.Dataset.from_generator(generator=lambda: self.dataset_generator(),
                                                      output_types=(tf.float32, tf.int32),
                                                      output_shapes=(
                                                          tf.TensorShape([None, None]), tf.TensorShape([None, None])))
            data_set = data_set.map(
                lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=self.classes, dtype=tf.float32)))
            data_set = data_set.map(lambda x, y: (tf.expand_dims(tf.cast(x, tf.float32), -1), y))
            data_set = data_set.map(lambda x, y: ({'image': x}, {'label': y}))
            if self.mode == 'train':
                data_set = data_set.batch(self.batch_size)
                data_set = data_set.repeat()
            if self.mode == 'eval':
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
