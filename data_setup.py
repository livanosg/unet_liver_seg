import os
import random
import zipfile
from glob import glob
from shutil import rmtree
import numpy as np
from cv2.cv2 import imwrite
from pydicom import dcmread
from config import paths, dataset_root


class DataHandler:
    """Handle CHAOS dataset from zip files placed in 'Dataset' folder."""

    def __init__(self):
        self.setup_paths = {'chaos_train': paths['train'] + '/CHAOS',
                            'chaos_eval': paths['eval'] + '/CHAOS',
                            'chaos_train_zip': dataset_root + '/CHAOS_Train_Sets.zip',
                            'chaos_test_zip': dataset_root + '/CHAOS_Test_Sets.zip'}

    def unzip_chaos(self):
        """Unzip CHAOS train and test dataset."""

        zip_paths = [self.setup_paths['chaos_train_zip'], self.setup_paths['chaos_test_zip']]
        for zip_path in zip_paths:
            print('Extracting {} ...'.format(zip_path))
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_root)
            print('Done!')
        os.replace(dataset_root + '/Train_Sets', self.setup_paths['chaos_train'])

        
    def set_eval(self):
        """Setup evaluation data-set. 2 random patients are chosen from CHAOS dataset."""
        for modality in ['/CT', '/MR']:
            print(modality[1:].center(10, ' ').center(100, '*'))
            for eval_sets in [self.setup_paths['chaos_eval']]:
                if os.path.exists(eval_sets + modality):
                    print('Removing patients from {} folder'.format(eval_sets[1:]).center(100, '-'))
                    eval_set = glob(eval_sets + modality + '/**')
                    for eval_patient in eval_set:
                        print('Moving: {}'.format(eval_patient).center(60, ' ').center(100, '|'))
                        print('to: {}'.format(eval_patient.replace('Eval', 'Train')).center(60, ' ').center(100, '|'))
                        os.makedirs(os.path.dirname(eval_patient.replace('Eval', 'Train')), exist_ok=True)
                        os.replace(eval_patient, eval_patient.replace('Eval', 'Train'))

            for train_set in [self.setup_paths['chaos_train']]:
                print('Moving patients from {} folder'.format(train_set.split('/')[-1]).center(100, '-'))
                patient_list = glob(train_set + modality + '/**')
                if patient_list:
                    eval_patients = random.sample(patient_list, k=2)
                    for patient in eval_patients:
                        print('Moving: {}'.format(patient).center(60, ' ').center(100, '|'))
                        print('to: {}'.format(patient.replace('Train', 'Eval')).center(60, ' ').center(100, '|'))
                        os.makedirs(os.path.dirname(patient.replace('Train', 'Eval')), exist_ok=True)
                        os.replace(patient, patient.replace('Train', 'Eval'))

    def setup_datasets(self):
        os.makedirs(paths['train'], exist_ok=True)
        os.makedirs(paths['eval'], exist_ok=True)
        self.unzip_chaos()
        self.set_eval()

    def reset_datasets(self):
        rmtree(paths['eval'])
        rmtree(paths['train'])
        rmtree(paths['chaos-test'])
        self.setup_datasets()


if __name__ == '__main__':
    a = DataHandler()
    a.setup_datasets()
