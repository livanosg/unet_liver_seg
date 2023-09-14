import os
import random
import zipfile
from glob import glob
import shutil
from config import TRAIN_DIR, EVAL_DIR, CHAOS_TRAIN_ZIP, CHAOS_TEST_ZIP, TEST_DIR, DATA_DIR


def unzip_chaos():
    """Unzip CHAOS train and test dataset."""
    for zip_path, dataset in [(CHAOS_TRAIN_ZIP, TRAIN_DIR), (CHAOS_TEST_ZIP, TEST_DIR)]:
        print('Extracting {} ...'.format(zip_path))
        if len(os.listdir(dataset)) == 0:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(path=DATA_DIR)
            print('Done!')
        else:
            print("{} is not empty!".format(dataset))


def set_eval():
    """Setup evaluation data-set. 2 random patients are chosen from CHAOS dataset."""

    def move(from_: str, to_, samples: int = None):
        """ Move patient samples from from_ to to_. If N samples provided, N samples will be moved."""
        if samples:
            patients = random.sample(glob(os.path.join(from_, '**')), k=samples)
        else:
            patients = glob(os.path.join(from_, '**'))
        for patient in patients:
            patient_to_ = patient.replace(from_, to_)
            print('Moving: {} patient from {} to {}'.format(os.path.basename(patient),
                                                            patient.split("/")[-3],
                                                            patient_to_.split("/")[-3]))
            if not os.path.exists(patient_to_):
                os.makedirs(os.path.dirname(patient_to_), exist_ok=True)
                os.rename(patient, patient_to_)
            else:
                print("Patient {} already in {}!!!".format(os.path.basename(patient_to_),
                                                           os.path.dirname(patient_to_)))

    for modality in ['CT', 'MR']:
        train_mod_dir = os.path.join(TRAIN_DIR, modality)
        eval_mod_dir = os.path.join(EVAL_DIR, modality)
        print(modality.center(6, ' ').center(47, '*'))
        if os.path.exists(eval_mod_dir):
            # Move patients from evaluation to train for resampling.
            move(from_=eval_mod_dir, to_=train_mod_dir)

        if os.path.exists(train_mod_dir):
            # Move patients from train to evaluation after sampling N samples.
            move(from_=train_mod_dir, to_=eval_mod_dir, samples=2)


def setup_datasets():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    unzip_chaos()
    set_eval()


def reset_datasets(self):
    shutil.rmtree(EVAL_DIR)
    shutil.rmtree(TRAIN_DIR)
    shutil.rmtree(TEST_DIR)
    self.setup_datasets()


if __name__ == '__main__':
    setup_datasets()
