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
    """Handle CHAOS and 3Dircadb datasets from zip files placed in 'Dataset' folder."""

    def __init__(self):
        self.setup_paths = {'ircadb_train': paths['train'] + '/IRCAD',
                            'ircadb_eval': paths['eval'] + '/IRCAD',
                            'chaos_train': paths['train'] + '/CHAOS',
                            'chaos_eval': paths['eval'] + '/CHAOS',
                            'chaos_train_zip': dataset_root + '/CHAOS_Train_Sets.zip',
                            'chaos_test_zip': dataset_root + '/CHAOS_Test_Sets.zip',
                            'ircadb_root': dataset_root + '/3Dircadb1'}

    def unzip_chaos(self):
        """Unzip CHAOS train and test dataset."""

        zip_paths = [self.setup_paths['chaos_train_zip'], self.setup_paths['chaos_test_zip']]
        for zip_path in zip_paths:
            print('Extracting {} ...'.format(zip_path))
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_root)
            print('Done!')
        os.replace(dataset_root + '/Train_Sets', self.setup_paths['chaos_train'])

    def unzip_ircadb(self):
        """Unzip 3Dircadb. Only healthy liver acquisitions are taken in consideration
        (3Dircadb1.5', '3Dircadb1.7', '3Dircadb1.11', '3Dircadb1.14', '3Dircadb1.20)."""

        conf_pat = {'file': 'PATIENT_DICOM.zip', 'zip_sub': 'PATIENT_DICOM/', 'to': 'DICOM_anon/'}
        conf_label = {'file': 'MASKS_DICOM.zip', 'zip_sub': 'MASKS_DICOM/liver/', 'to': 'Ground_dcm/'}
        pat = ['3Dircadb1.5', '3Dircadb1.7', '3Dircadb1.11', '3Dircadb1.14', '3Dircadb1.20']
        with zipfile.ZipFile(self.setup_paths['ircadb_root'] + '.zip', 'r') as zip_ref:
            for i in zip_ref.infolist():
                if os.path.dirname(i.filename) in pat:
                    zip_ref.extract(i.filename, self.setup_paths['ircadb_root'])
        for conf in [conf_pat, conf_label]:
            for dirpath, dirnames, filenames in os.walk(self.setup_paths['ircadb_root']):
                for file in filenames:
                    if (os.path.basename(dirpath) in pat) and file == conf['file']:
                        filepath = os.path.join(dirpath, file)
                        print('Extracting {}'.format(filepath))
                        with zipfile.ZipFile(filepath, 'r') as zip_ref:
                            liver_list = [liver_path for liver_path in zip_ref.namelist() if
                                          conf['zip_sub'] in liver_path]
                            for liver_path in liver_list:
                                filepath = os.path.join(dirpath, liver_path)
                                zip_ref.extract(member=zip_ref.getinfo(liver_path), path=dirpath)
                                if os.path.isfile(filepath):
                                    os.makedirs(os.path.dirname(filepath.replace(conf['zip_sub'], conf['to']) + '.dcm'),
                                                exist_ok=True)
                                    os.replace(filepath, filepath.replace(conf['zip_sub'], conf['to']) + '.dcm')
                            os.removedirs(os.path.dirname(filepath))

    def rename_ircadb(self):
        """Set ircadb in CHAOS dataset format."""

        print('Splitting dataset from main folder...')
        patient = sorted(glob(self.setup_paths['ircadb_root'] + '/**'))
        for i, patient_id in enumerate(patient):
            print(patient_id)
            # if patient_id in ('3Dircadb1.5', '3Dircadb1.7', '3Dircadb1.11', '3Dircadb1.14', '3Dircadb1.20'):
            os.replace(patient_id, patient_id.replace(os.path.basename(patient_id), str(i + 1)))
        patient_2 = sorted(glob(self.setup_paths['ircadb_root'] + '/**/**'))
        for i in patient_2:
            if 'DICOM_anon' in i or 'Ground_dcm' in i:
                os.makedirs(os.path.dirname(
                    i.replace(self.setup_paths['ircadb_root'], self.setup_paths['ircadb_train'] + '/CT')),
                    exist_ok=True)
                os.replace(i, i.replace(self.setup_paths['ircadb_root'], self.setup_paths['ircadb_train'] + '/CT'))
        print('Removing Extracted Folder...')
        rmtree(self.setup_paths['ircadb_root'])
        print('Done!')

    def make_png_ircadb(self):
        """Save labels in .png format."""

        ground_dcm = sorted(glob(self.setup_paths['ircadb_train'] + '/**/Ground_dcm/**.dcm', recursive=True))
        print('Converting dcm labels to png...')
        for ground_dcm_path in ground_dcm:
            ground_pxl = dcmread(ground_dcm_path).pixel_array
            png_path = ground_dcm_path.replace('Ground_dcm', 'Ground').replace('.dcm', '.png')
            max_val = np.max(ground_pxl)
            if max_val > 0:
                ground_pxl[ground_pxl == max_val] = 255
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            imwrite(png_path, ground_pxl)
        print('Removing dcm labels...')
        ground_dcm_folder_list = sorted(glob(self.setup_paths['ircadb_train'] + '/**/Ground_dcm', recursive=True))
        for ground_dcm_folder in ground_dcm_folder_list:
            rmtree(ground_dcm_folder)
        print('Done!')

    def set_eval(self):
        """Setup evaluation data-set. 2 random patients are chosen from CHAOS dataset while 1 random patient is taken
        from Ircadb."""
        for modality in ['/CT', '/MR']:
            print(modality[1:].center(10, ' ').center(100, '*'))
            for eval_sets in [self.setup_paths['ircadb_eval'], self.setup_paths['chaos_eval']]:
                if os.path.exists(eval_sets + modality):
                    print('Removing patients from {} folder'.format(eval_sets[1:]).center(100, '-'))
                    eval_set = glob(eval_sets + modality + '/**')
                    for eval_patient in eval_set:
                        print('Moving: {}'.format(eval_patient).center(60, ' ').center(100, '|'))
                        print('to: {}'.format(eval_patient.replace('Eval', 'Train')).center(60, ' ').center(100, '|'))
                        os.makedirs(os.path.dirname(eval_patient.replace('Eval', 'Train')), exist_ok=True)
                        os.replace(eval_patient, eval_patient.replace('Eval', 'Train'))

            for train_set in [self.setup_paths['chaos_train'], self.setup_paths['ircadb_train']]:
                print('Moving patients from {} folder'.format(train_set.split('/')[-1]).center(100, '-'))
                patient_list = glob(train_set + modality + '/**')
                if patient_list:
                    if '/CHAOS' in train_set:
                        eval_patients = random.sample(patient_list, k=2)
                    else:
                        eval_patients = random.sample(patient_list, k=1)
                    for patient in eval_patients:
                        print('Moving: {}'.format(patient).center(60, ' ').center(100, '|'))
                        print('to: {}'.format(patient.replace('Train', 'Eval')).center(60, ' ').center(100, '|'))
                        os.makedirs(os.path.dirname(patient.replace('Train', 'Eval')), exist_ok=True)
                        os.replace(patient, patient.replace('Train', 'Eval'))

    def setup_datasets(self):
        os.makedirs(paths['train'], exist_ok=True)
        os.makedirs(paths['eval'], exist_ok=True)
        self.unzip_ircadb()
        self.rename_ircadb()
        self.make_png_ircadb()
        self.unzip_chaos()
        self.set_eval()

    def reset_datasets(self):
        rmtree(paths['eval'])
        rmtree(paths['train'])
        rmtree(paths['chaos-test'])
        self.setup_datasets()


if __name__ == '__main__':
    a = DataHandler()
    a.set_eval()
